"""
Modified from https://github.com/thu-ml/tianshou/blob/master/tianshou/env/venvs.py
"""
from distutils.log import info
import numpy as np, time
from pyrl.utils.data import DictArray, GDict, SharedDictArray, split_list_of_parameters, concat, is_num, decode_np
from pyrl.utils.meta import Worker
from pyrl.utils.math import split_num
from .env_utils import build_single_env, get_max_episode_steps


class VectorEnv:
    def __init__(self, env_cfgs=None, use_render=False, wait_num=None, timeout=None, shared_memory=False, **kwargs):
        """
        env_cfgs can contain different environments, but the format of information should be the same.
        using rendering system in sapien will enlarge the memory usage, so we disable it by default.
        Please use shared_memory when the environment is very fast.
        """
        assert wait_num is None and timeout is None, "We do not support now!"
        self.use_render = use_render
        self.timeout = int(1e9) if timeout is None else timeout
        self.wait_num = len(env_cfgs) if wait_num is None and env_cfgs is not None else wait_num

        self.sub_env = False
        self.shared_memory = shared_memory
        self.dirty = False

        if env_cfgs is None:
            self.sub_env = True
            return
        self.num_envs = len(env_cfgs)
        example_env = build_single_env(env_cfgs[0])
        self._max_episode_steps = get_max_episode_steps(example_env)
        self.example_env = example_env
        self.iscost = example_env.iscost
        if shared_memory:
            item = {
                "obs": example_env.reset(),
                "actions": example_env.random_action(),
            }
            item["next_obs"] = item["obs"]
            item["dones"] = True
            item["episode_dones"] = True
            item["rewards"] = np.float32(1.0)
            item["infos"] = np.zeros(1, dtype=np.dtype("U10000"))  # 10KB for every infos

            self.use_render = use_render
            if use_render:
                item["images"] = example_env.render("rgb_array")
            self.infos = SharedDictArray(DictArray(GDict(item).to_array(), capacity=self.num_envs))
            mem_infos = self.infos.get_infos()
        else:
            self.infos = mem_infos = None
        self.episode_dones = np.zeros([self.num_envs, 1], dtype=np.bool)
        self.recent_obs = DictArray(GDict(example_env.reset()).to_array(), capacity=self.num_envs)
        base_seed = np.random.randint(int(1e9))
        self.workers = [Worker(build_single_env, i, base_seed, True, mem_infos, cfg=cfg) for i, cfg in enumerate(env_cfgs)]

    def get_sub_env(self, num):
        # Use the shared worker to create a new env, it is designed for on-policy algorithm evaluation
        assert num <= self.num_envs
        env = VectorEnv(None, self.use_render, None, None, self.shared_memory)
        env.timeout = self.timeout
        env.wait_num = self.wait_num
        env.num_envs = num
        env._max_episode_steps = self._max_episode_steps
        env.example_env = self.example_env
        env.iscost = self.iscost
        env.infos = self.infos.to_dict_array().take(slice(0, num)) if self.shared_memory else None
        env.episode_dones = self.episode_dones[:num]
        env.recent_obs = self.recent_obs.take(slice(0, num))
        env.workers = self.workers[:num]
        env.sub_env = True
        env.reset()
        return env

    @property
    def running_idx(self):
        return [i for i, env in enumerate(self.workers) if env.is_running]

    @property
    def ready_idx(self):
        return [i for i, env in enumerate(self.workers) if env.is_ready]

    @property
    def idle_idx(self):
        return [i for i, env in enumerate(self.workers) if env.is_idle]

    @property
    def done_idx(self):
        return [i for i, env in enumerate(self.workers) if env.is_idle and self.episode_dones[i, 0]]

    def _assert_id(self, idx) -> None:
        idx = list(range(self.num_envs)) if idx is None else idx
        assert isinstance(idx, (list, tuple))
        for i in idx:
            # print(self.workers[i].is_running, self.workers[i].item_in_pipe.value)
            assert self.workers[i].is_idle, f"Cannot interact with environment {i} which is stepping now."
        return idx

    def random_action(self):
        return np.stack([self.example_env.action_space.sample() for i in range(self.num_envs)], axis=0)

    def _get_results(self, keys=None, idx=None, sync=True, to_list=False, mode="reset", encode_info=True):
        idx = range(self.num_envs) if idx is None else idx
        if sync:
            ret = [self.workers[i].get() for i in idx]
            if self.shared_memory:
                ret = self.infos.to_dict_array().select_by_keys(keys, to_list).take(idx, wrapper=False)
            else:
                ret = DictArray.stack(ret, axis=0).select_by_keys(keys, to_list, wrapper=False)
            res = ret
        else:
            finish_index, finish_res = [], []
            st = time.time()
            while len(idx) > 0:
                new_idx = []
                for i in idx:
                    ret_i = self.workers[i].get_async()
                    if ret_i is not None:
                        finish_index.append(i)
                        if not self.shared_memory:
                            finish_res.append(ret_i["obs"])
                    else:
                        new_idx.append(i)
                idx = new_idx
                if not sync or len(finish_index) >= self.wait_num or time.time() - st >= self.timeout:
                    break
            if len(finish_index) == 0:
                return None
            if self.shared_memory:
                ret = self.infos.to_dict_array().select_by_keys(keys, to_list).take(finish_index, wrapper=False)
            else:
                ret = DictArray.stack(finish_res, axis=0).select_by_keys(keys, to_list, wrapper=False)
            idx = finish_index
            ret, res = [ret, idx], ret
        if mode == "reset":
            assert to_list
            self.recent_obs.assign(idx, res)
            self.episode_dones[idx] = 0
        else:
            assert mode == "step"
            self.recent_obs.assign(idx, res[0] if to_list else res["next_obs"])
            self.episode_dones[idx] = res[2] if to_list else res["episode_dones"]
            # print(ret['infos'], encode_info)
            if "infos" in ret and not encode_info:
                infos = [decode_np(item[0]) for item in ret["infos"]]
                infos = GDict(infos).to_array(wrapper=False)
                ret["infos"] = DictArray.stack(infos, axis=0, wrapper=True).f64_to_f32(wrapper=False)
        return ret

    def reset(self, level=None, idx=None):
        self.dirty = False
        idx = self._assert_id(idx)
        if level is not None:
            assert is_num(level) or len(level) == len(idx)
        assert self._assert_id(idx)
        for i in idx:
            kwargs = {} if level is None else dict(level=(level if is_num(level) else level[i]))
            self.workers[i].call("reset_dict", **kwargs)
        return self._get_results("obs", idx, True, True, "reset")

    def step(self, actions, with_info=False, idx=None, sync=True, to_list=True, encode_info=True):
        assert not self.dirty
        idx = self._assert_id(idx)
        assert len(actions) == len(idx), f"{len(actions)} {len(idx)}"
        for i in idx:
            self.workers[i].call("step_dict", action=actions[i], with_info=with_info, encode_info=True)
        if sync:
            if to_list:
                # The original gym interface.
                keys = ["next_obs", "rewards", "episode_dones"]
            else:
                keys = ["obs", "actions", "next_obs", "rewards", "dones", "episode_dones"]
            if with_info:
                keys += ["infos"]
            ret = self._get_results(keys, idx, True, to_list, "step", encode_info=encode_info)
            return ret

    def render(self, mode="rgb_array", idx=None):
        assert self.use_render
        idx = self._assert_id(idx)
        for i in idx:
            self.workers[i].call("render_dict", mode=mode)
        for i in idx:
            self.workers[i].get()
        return self.infos["images"][idx]

    def step_random_actions(self, num, with_info=False, encode_info=True):
        # For replay buffer warmup of the RL agent
        idx = self._assert_id(None)
        self.dirty = True

        n, nums = split_num(num, len(idx))
        idx = idx[:n]
        shared_mem_value = []
        for i in idx:
            shared_mem_value.append(self.workers[i].shared_memory.value)
            self.workers[i].set_shared_memory(False)
            self.workers[i].call("step_random_actions", num=nums[i], with_info=with_info, encode_info=encode_info)
        ret = []
        for i in idx:
            ret_i = self.workers[i].get()
            ret_i["worker_index"] = np.ones(ret_i["dones"].shape, dtype=np.int16) * i
            ret.append(ret_i)
            self.workers[i].set_shared_memory(shared_mem_value[i])
        ret = DictArray.concat(ret, axis=0, wrapper=False)
        return ret

    def __del__(self):
        if self.sub_env:
            return
        for worker in self.workers:
            worker.terminate()
            del worker
