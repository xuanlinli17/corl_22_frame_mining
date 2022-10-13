import time

import numpy as np
from pyrl.utils.data import DictArray, GDict, to_np
from pyrl.utils.meta import get_logger

from .builder import ROLLOUTS
from .env_utils import build_env


def set_rnn_states(states, index, value=0):
    if index is None:
        index = slice(None)
    if isinstance(states, (list, tuple)):
        if isinstance(value, (list, tuple)):
            for _, __ in zip(states, value):
                _[index] = __[index]
        else:
            for _ in states:
                _[index] = value
    elif states is not None:
        states[index] = value


def take_recurrent_state(states, index):
    if index is None:
        index = slice(None)
    if isinstance(states, (list, tuple)):
        return [_[index].clone() for _ in states]
    elif states is not None:
        return states[index].clone()
    return None


@ROLLOUTS.register_module()
class Rollout:
    def __init__(self, env_cfg, num_procs=20, sync=True, seeds=None, single_procs=True, with_info=False, **kwargs):
        self.env = build_env(env_cfg, num_procs, single_procs=single_procs, **kwargs)
        self.with_info = with_info
        self.n = self.env.num_envs
        self.seeds = seeds
        self.sync = sync
        self.rnn_states = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, idx=None, *args, **kwargs):
        if idx is not None:
            kwargs = dict(kwargs)
            kwargs["idx"] = idx
        set_rnn_states(self.rnn_states, idx, 0)
        return self.env.reset(*args, **kwargs)

    @property
    def recent_obs(self):
        return self.env.recent_obs if self.vector_env else GDict(self.env.recent_obs).unsqueeze(0, False)

    def forward_with_policy(self, pi=None, num=1, on_policy=False, replay=None):
        if pi is None:
            assert not on_policy
            return self.env.step_random_actions(num, with_info=False, encode_info=False)

        sim_time, agent_time, oh_time = 0, 0, 0

        import torch

        def get_actions(idx=None, with_states=False):
            done_index = self.env.done_idx
            if len(done_index) > 0:
                self.reset(idx=done_index)

            obs = DictArray(self.recent_obs).take(idx, wrapper=False) if idx is not None else self.recent_obs
            with torch.no_grad():
                actions, next_rnn_states = pi(obs, rnn_states=take_recurrent_state(self.rnn_states, idx), rnn_mode="with_states")
                actions = to_np(actions)
                if next_rnn_states is not None:
                    if self.rnn_states is None:
                        if isinstance(next_rnn_states, (tuple, list)):
                            self.rnn_states = [torch.repeat_interleave(_[:1] * 0, self.n, 0) for _ in next_rnn_states]
                        else:
                            self.rnn_states = torch.repeat_interleave(next_rnn_states[:1] * 0, self.n, 0)
                    current_states = take_recurrent_state(self.rnn_states, idx)
                    set_rnn_states(self.rnn_states, idx, next_rnn_states)
                else:
                    current_states = None
            return (actions, current_states, next_rnn_states) if with_states else actions

        sync = self.sync or on_policy or num != self.n or not self.is_vec_env
        if not sync:
            assert self.is_vec_env, "Only multi-process env supports asynchorized sample collection!"
            idx = self.env.idle_idx
            if len(idx) > 0:
                ret = self.env._get_results(
                    keys=["obs", "next_obs", "actions", "rewards", "dones", "episode_dones", "infos"],
                    sync=False,
                    mode="step",
                    idx=idx,
                    encode_info=False,
                )
                if ret is not None:
                    ret["worker_index"] = np.array(idx, dtype=np.int32).reshape(-1, 1)
            else:
                ret = None
            self.env.step(get_actions(idx)[0], with_info=self.with_info, idx=idx, sync=False, to_list=False)
        elif on_policy:
            trajs = [[] for i in range(self.env.num_envs)]
            total, unfinished, finished, ret = 0, 0, 0, None
            true_array = np.ones(1, dtype=np.bool)
            last_get_done = time.time()
            while total < num:
                st = time.time()
                actions, rnn_states, next_rnn_states = get_actions(with_states=True)
                agent_time += time.time() - st

                st = time.time()
                item = self.env.step(actions, with_info=self.with_info, idx=None, sync=True, to_list=False, encode_info=False)
                sim_time += time.time() - st

                st = time.time()
                unfinished += self.n
                total += self.n
                item["worker_index"] = np.arange(self.n, dtype=np.int32)[:, None]
                item["is_truncated"] = np.zeros(self.n, dtype=np.bool)[:, None]
                if self.rnn_states is not None:
                    item["rnn_states"] = GDict(rnn_states).to_numpy(wrapper=False)
                    item["next_rnn_states"] = GDict(next_rnn_states).to_numpy(wrapper=False)
                item = DictArray(item)
                for i in range(self.n):
                    item_i = item.take(i)
                    trajs[i].append(item_i)
                    if item_i["episode_dones"][0]:
                        unfinished -= len(trajs[i])
                        if len(trajs[i]) + finished > num:
                            trajs[i] = trajs[i][: num - finished]
                        replay.push_batch(DictArray.stack(trajs[i], axis=0, wrapper=False))
                        finished += len(trajs[i])
                        trajs[i] = []

                oh_time += time.time() - st
            st = time.time()
            if unfinished > 0:
                for i in range(self.n):
                    if len(trajs[i]) > 0 and finished < num:
                        if len(trajs[i]) + finished > num:
                            trajs[i] = trajs[i][: num - finished]
                        trajs[i][-1]["is_truncated"] = true_array
                        replay.push_batch(DictArray.stack(trajs[i], axis=0, wrapper=False))
                        finished += len(trajs[i])
                del trajs

            oh_time += time.time() - st
            get_logger().info(
                f"Finish with {finished} samples, simulation time/FPS:{sim_time:.2f}/{finished / sim_time:.2f}, agent time/FPS:{agent_time:.2f}/{finished / agent_time:.2f}, overhead time:{oh_time:.2f}"
            )
        else:
            assert num % self.n == 0, f"{self.n} % {num} != 0, some processes may be idel!"
            ret = []
            for i in range(num // self.n):
                action = get_actions()
                item = self.env.step(action, with_info=False, idx=None, sync=True, to_list=False, encode_info=False)
                item["worker_index"] = np.arange(self.n, dtype=np.int32)[:, None]
                ret.append(item)
            ret = DictArray.concat(ret, axis=0, wrapper=False)
        if ret is not None:
            assert not on_policy
            replay.push_batch(ret)
        else:
            ret = replay.get_all().memory
        return ret

    def close(self):
        del self.env

