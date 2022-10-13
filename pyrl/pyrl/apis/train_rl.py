import itertools
import os
import os.path as osp
import time
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
from pyrl.env import ReplayMemory, save_eval_statistics
from pyrl.utils.data import GDict, dict_to_str, is_not_null, num_to_str
from pyrl.utils.math import EveryNSteps
from pyrl.utils.meta import get_logger, get_total_memory, td_format
from pyrl.utils.torch import TensorboardLogger, save_checkpoint


class EpisodicStatistics:
    def __init__(self, num_procs, info_keys_mode={}):
        self.num_procs = num_procs
        self.info_keys_mode = OrderedDict(
            {"rewards": [True, "sum", "all"], "lens": [True, "sum", "all"]}
        ) 
        self.info_keys_mode.update(info_keys_mode)
        for key, item in self.info_keys_mode.items():
            assert item[2] in ["mean", "all"]
            assert item[1] in ["mean", "min", "max", "sum"]

        self.num_workers = 1
        self.reset_current()
        self.reset_history()

    def push(self, trajs):
        rewards, dones, index, infos = trajs["rewards"], trajs["episode_dones"], trajs.get("worker_index", None), trajs.get("infos", None)
        self.expand_buffer(index)
        for j in range(len(rewards)):
            i = 0 if index is None else int(index[j])
            self.current[i]["lens"] += 1
            self.current[i]["rewards"] += float(rewards[j])
            if dones[j]:
                for key, value in self.current[i].items():
                    if key not in ["rewards", "lens"] and self.info_keys_mode[key][1] == "mean":
                        value /= self.current[i]["lens"]
                    self.history[key].append(value)
                self.current[i] = defaultdict(float)

    def expand_buffer(self, index):
        max_index = np.max(index) + 1
        for i in range(max(max_index - self.num_workers, 0)):
            self.current.append(defaultdict(float))

    def reset_history(self):
        self.history = defaultdict(list)

    def reset_current(self):
        self.current = [
            defaultdict(float),
        ]

    def get_sync_stats(self):
        num_ep = len(self.history["rewards"])
        history_min = {key: np.min(value) for key, value in self.history.items()}
        history_max = {key: np.max(value) for key, value in self.history.items()}
        history_sum = {key: np.sum(value) for key, value in self.history.items()}
        history_mean = {key: item / num_ep for key, item in history_sum.items()}
        return history_min, history_max, history_mean

    def get_stats_str(self):
        history_min, history_max, history_mean = self.get_sync_stats()
        ret = ""
        for key, item in self.info_keys_mode.items():
            if not (key in history_mean and item[0]):
                continue
            if len(ret) > 0:
                ret += ", "
            if key == "lens":
                precision = 0
            elif key == "rewards":
                precision = 1
            else:
                precision = 2
            mean_i = num_to_str(history_mean[key], precision=precision)
            min_i = num_to_str(history_min[key], precision=precision)
            max_i = num_to_str(history_max[key], precision=precision)

            ret += f"{key}:{mean_i}"
            if item[2] == "all":
                ret += f"[{min_i}, {max_i}]"
        return ret

    def get_stats(self):
        history_min, history_max, history_mean = self.get_sync_stats()
        ret = {}
        for key in self.info_keys_mode:
            if key in history_mean:
                ret[f"env/{key}_mean"] = history_mean[key]
                ret[f"env/{key}_min"] = history_min[key]
                ret[f"env/{key}_max"] = history_max[key]
        return ret


def train_rl(
    agent,
    rollout,
    evaluator,
    replay,
    on_policy,
    work_dir,
    expert_replay=None,
    recent_traj_replay=None,
    total_steps=1000000,
    warm_steps=10000,
    resume_steps=0,
    use_policy_to_warm_up=False,
    print_steps=None,
    n_steps=1,
    n_updates=1,
    n_checkpoint=None,
    n_eval=None,
    eval_cfg=None,
    ep_stats_cfg={},
    warm_up_training=False,
    warm_up_train_q_only=-1,
):
    logger = get_logger()
    agent.set_mode(mode="train")

    import torch
    from pyrl.utils.torch import get_cuda_info

    tf_logs = ReplayMemory(n_updates, None)
    tf_logs.reset()
    tf_logger = TensorboardLogger(work_dir)

    if print_steps is None:
        print_steps = n_steps

    checkpoint_dir = osp.join(work_dir, "models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rollout is not None:
        obs = rollout.reset()
        agent.reset()
        logger.info(f"Rollout state dim: {GDict(obs).shape}, action dim: {GDict(rollout.random_action()).shape}!")
        episode_statistics = EpisodicStatistics(rollout.n, **ep_stats_cfg)
        total_episodes = 0
    else:
        # Batch RL
        all_shape = replay.memory.shape
        logger.info(f"Loaded replay buffer shape: {all_shape}!")

    check_eval = EveryNSteps(n_eval)
    if is_not_null(n_eval) and (n_checkpoint is None or n_checkpoint > n_eval):
        n_checkpoint = n_eval

    check_checkpoint = EveryNSteps(n_checkpoint)
    check_tf_log = EveryNSteps(1000)

    total_updates = 0
    print_replay_shape = False

    if warm_steps > 0:
        logger.info(f"Begin {warm_steps} warm-up steps!")
        # Randomly warm up replay buffer for model-free RL and learned model for mode-based RL
        assert not on_policy
        assert rollout is not None
        trajectories = rollout.forward_with_policy(agent if use_policy_to_warm_up else None, warm_steps)
        # print(GDict(trajectories).shape)
        # exit(0)
        episode_statistics.push(trajectories)
        replay.push_batch(trajectories)

        rollout.reset()
        agent.reset()
        episode_statistics.reset_current()

        check_eval.check(warm_steps)
        check_checkpoint.check(warm_steps)
        check_tf_log.check(warm_steps)
        logger.info(f"Finish {warm_steps} warm-up steps!")
        if warm_up_train_q_only > 0:
            for i in range(warm_up_train_q_only):
                total_updates += 1
                training_infos = agent.update_parameters(replay, updates=total_updates, q_only=True)
                logger.info(f"Warmup pretrain q network: {i}/{warm_up_train_q_only} {dict_to_str(training_infos)}")
        if warm_up_training:
            tf_logs.reset()
            for i in range(n_updates):
                total_updates += 1
                training_infos = GDict(agent.update_parameters(replay, updates=total_updates)).to_numpy()
                tf_logs.push(training_infos)

    steps = warm_steps + resume_steps
    total_steps += resume_steps
    begin_steps = steps

    begin_time = datetime.now()
    max_ETA_len = None
    logger.info("Begin training!")

    for iteration_id in itertools.count(1):
        if steps >= total_steps:
            break

        if rollout is not None:
            # For online RL algorithm
            episode_statistics.reset_history()
            if on_policy:
                replay.reset()
                rollout.reset()
                agent.reset()
                episode_statistics.reset_current()

        tb_print = defaultdict(list)
        tb_log = {}
        print_log = {}

        update_time = 0
        time_begin_episode = time.time()
        tmp_steps = 0

        if n_steps > 0:
            # For online RL
            collect_sample_time = 0
            num_episodes = 0
            num_done = 0
            if recent_traj_replay is not None:
                recent_traj_replay.reset()
            """
            For on-policy algorithm, we will print training infos for every gradient batch.
            For off-policy algorithm, we will print training infos for every n_steps epochs.
            """
            while num_done < print_steps and not (on_policy and num_done > 0):
                # Collect samples
                tmp_time = time.time()
                agent.eval()  # For things like batch norm
                trajectories = rollout.forward_with_policy(agent, n_steps, on_policy, replay)
                if not print_replay_shape:
                    print_replay_shape = True
                    logger.info(f"Replay buffer shape: {replay.memory.shape}")
                agent.train()

                if trajectories is not None:
                    if recent_traj_replay is not None:
                        recent_traj_replay.push_batch(trajectories)
                    episode_statistics.push(trajectories)
                    n_ep = np.sum(trajectories["episode_dones"].astype(np.int32))
                    num_done += n_ep
                    num_episodes += n_ep
                    tmp_steps += len(trajectories["rewards"])
                collect_sample_time += time.time() - tmp_time

                # Train agent
                for i in range(n_updates):
                    total_updates += 1
                    tmp_time = time.time()
                    extra_args = {} if expert_replay is None else dict(expert_replay=expert_replay)
                    training_infos = agent.update_parameters(replay, updates=total_updates, **extra_args)
                    # torch.cuda.empty_cache()

                    for key in training_infos:
                        tb_print[key].append(training_infos[key])
                    update_time += time.time() - tmp_time

                if hasattr(agent, "update_discriminator"):
                    assert recent_traj_replay is not None
                    tmp_time = time.time()
                    disc_update_applied = agent.update_discriminator(expert_replay, recent_traj_replay, n_ep)
                    if disc_update_applied:
                        recent_traj_replay.reset()
                    update_time += time.time() - tmp_time
            tb_print = {key: np.mean(tb_print[key]) for key in tb_print}

            ep_stats = episode_statistics.get_stats()

            episode_time, collect_sample_time = [time.time() - time_begin_episode, collect_sample_time]
            tmp_steps, num_episodes = [tmp_steps, int(num_episodes)]

            steps += tmp_steps
            total_episodes += num_episodes

            print_log["samples_stats"] = episode_statistics.get_stats_str()
            tb_print.update(dict(episode_time=episode_time, collect_sample_time=collect_sample_time))

            tb_log.update(ep_stats)
            tb_log.update(dict(num_episodes=num_episodes, total_episodes=total_episodes))
        else:
            # For offline RL
            for i in range(n_updates):
                total_updates += 1
                steps += 1

                tmp_time = time.time()
                training_infos = agent.update_parameters(replay, updates=total_updates)
                update_time += time.time() - tmp_time

                for key in training_infos:
                    tb_print[key].append(training_infos[key])
            tb_print = {key: np.mean(tb_print[key]) for key in tb_print}

        tb_log["update_time"] = update_time
        tb_log["total_updates"] = int(total_updates)
        tb_log["buffer_size"] = len(replay)

        tb_print, tb_log = [tb_print, tb_log]
        total_memory = get_total_memory("G", False)
        tb_print["memory"] = total_memory
        print_log.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))


        tb_log.update(tb_print)
        print_log.update(tb_print)
        print_log = {key.split("/")[-1]: val for key, val in print_log.items()}

        if check_tf_log.check(steps):
            # print(GDict(tb_log).shape, GDict(tb_log).type)
            tf_logger.log(tb_log, n_iter=steps, tag_name="train")
            # exit(0)

        percentage = f"{((steps - begin_steps) / (total_steps - begin_steps)) * 100:.0f}%"
        passed_time = td_format(datetime.now() - begin_time)
        ETA = td_format((datetime.now() - begin_time) * max((total_steps - begin_steps) / (steps - begin_steps) - 1, 0))
        if max_ETA_len is None:
            max_ETA_len = len(ETA)

        logger.info(f"{steps}/{total_steps}({percentage}) Passed time:{passed_time} ETA:{ETA} {dict_to_str(print_log)}")

        if check_eval.check(steps) and is_not_null(evaluator):
            standardized_eval_step = check_eval.standard(steps)
            logger.info(f"Begin to evaluate at step: {steps}. " f"The evaluation info will be saved at eval_{standardized_eval_step}")
            eval_dir = osp.join(work_dir, f"eval_{standardized_eval_step}")

            agent.eval()  # For things like batch norm
            agent.set_mode(mode="test")  # For things like obs normalization

            lens, rewards, finishes = evaluator.run(agent, **eval_cfg, work_dir=eval_dir)
            # agent.recover_data_parallel()

            torch.cuda.empty_cache()
            save_eval_statistics(eval_dir, lens, rewards, finishes)
            agent.train()
            agent.set_mode(mode="train")

            eval_dict = dict(mean_length=np.mean(lens), std_length=np.std(lens), mean_reward=np.mean(rewards), std_reward=np.std(rewards))
            tf_logger.log(eval_dict, n_iter=steps, tag_name="test")

        if check_checkpoint.check(steps):
            standardized_ckpt_step = check_checkpoint.standard(steps)
            model_path = osp.join(checkpoint_dir, f"model_{standardized_ckpt_step}.ckpt")
            logger.info(f"Save model at step: {steps}. The model will be saved at {model_path}")
            save_checkpoint(agent, model_path)

    if n_checkpoint:
        model_path = osp.join(checkpoint_dir, f"model_final.ckpt")
        logger.info(f"Save checkpoint at final step {total_steps}. The model will be saved at {model_path}.")
        agent.to_normal()
        save_checkpoint(agent, model_path)
        agent.recover_ddp()
