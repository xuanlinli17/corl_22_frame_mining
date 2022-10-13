import copy
import glob
import logging
import os
import os.path as osp
import shutil
from copy import deepcopy

import cv2
import numpy as np
from h5py import File
from pyrl.utils.data import DictArray, GDict, concat_list, decode_np, dict_to_str, is_str, num_to_str, split_list_of_parameters, to_np
from pyrl.utils.file import dump, load
from pyrl.utils.meta import get_logger, get_logger_name, get_total_memory

from .builder import EVALUATIONS
from .env_utils import build_env, build_single_env, get_env_state, true_done
from .replay_buffer import ReplayMemory


def save_eval_statistics(folder, lengths, rewards, finishes, logger=None):
    if logger is None:
        logger = get_logger()
    logger.info(
        f"Num of trails: {len(lengths):.2f}, "
        f"Length: {np.mean(lengths):.2f}\u00B1{np.std(lengths):.2f}, "
        f"Reward: {np.mean(rewards):.2f}\u00B1{np.std(rewards):.2f}, "
        f"Success or Early Stop Rate: {np.mean(finishes):.2f}\u00B1{np.std(finishes):.2f}"
    )
    if folder is not None:
        table = [["length", "reward", "finish"]]
        table += [[num_to_str(__, precision=2) for __ in _] for _ in zip(lengths, rewards, finishes)]
        dump(table, osp.join(folder, "statistics.csv"))



@EVALUATIONS.register_module()
class Evaluation:
    def __init__(
        self,
        env_cfg,
        worker_id=None,
        save_traj=True,
        only_save_success_traj=False,
        save_video=True,
        save_scene_state=False,
        use_log=True,
        log_every_step=False,
        log_all=False,
        horizon=None,
        sample_mode="eval",
        eval_init_levels=None,
        **kwargs,
    ):

        env_cfg = copy.deepcopy(env_cfg)
        env_cfg["unwrapped"] = False
        if horizon is not None:
            env_cfg["horizon"] = horizon
        self.env = build_single_env(env_cfg)
        # print(self.env)
        # exit(0)
        self.env.reset()

        if hasattr(self.env, "_max_episode_steps"):
            horizon = self.env._max_episode_steps
        elif hasattr(self.env.unwrapped, "_max_episode_steps"):
            horizon = self.env.unwrapped._max_episode_steps
        else:
            print("No env horizon!")
            exit(0)

        self.horizon = horizon
        # print(self.horizon, env_cfg)
        # exit(0)
        self.save_traj = save_traj
        self.only_save_success_traj = only_save_success_traj
        self.save_video = save_video
        self.env_name = env_cfg.env_name
        self.worker_id = worker_id

        logger_name = get_logger_name()
        log_level = logging.INFO if (log_all or self.worker_id is None or self.worker_id == 0) else logging.ERROR
        worker_suffix = "-env" if self.worker_id is None else f"-env-{self.worker_id}"

        self.logger = get_logger("Evaluation-" + logger_name + worker_suffix, with_stream=use_log, log_level=log_level)
        self.log_every_step = log_every_step

        self.save_scene_state = save_scene_state
        self.sample_mode = sample_mode

        self.work_dir, self.video_dir, self.trajectory_path = None, None, None
        self.h5_file = None

        self.episode_id = 0
        self.level_index = 0
        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        self.recent_obs = None

        self.data_episode = None
        self.video_writer = None
        self.video_file = None

        # restrict the levels as those randomly sampled from eval_init_levels_path, if eval_init_levels_path is not None
        if eval_init_levels is not None:
            if is_str(eval_init_levels):
                is_csv = eval_init_levels.split(".")[-1] == "csv"
                eval_init_levels = load(eval_init_levels)
                if is_csv:
                    eval_init_levels = eval_init_levels[0]
            self.eval_init_levels = eval_init_levels
            self.logger.info(f"During evaluation, levels are selected from an existing list with length {len(self.eval_init_levels)}")
        else:
            self.eval_init_levels = None

        if save_video:
            # Use rendering with use additional 1Gi memory in sapien
            image = self.env.render("rgb_array")
            self.logger.info(f"Size of image in the rendered video {image.shape}")

        if hasattr(self.env, "seed") and self.worker_id is not None:
            # Make sure that envs in different processes have different behaviors
            seed = np.random.randint(0, int(1e8))
            self.logger.info(f"Set environment seed {seed}")
            self.env.seed(seed)

    def start(self, work_dir=None):
        if work_dir is not None:
            self.work_dir = work_dir if self.worker_id is None else os.path.join(work_dir, f"thread_{self.worker_id}")
            # shutil.rmtree(self.work_dir, ignore_errors=True)
            os.makedirs(self.work_dir, exist_ok=True)
            if self.save_video:
                self.video_dir = osp.join(self.work_dir, "videos")
                os.makedirs(self.video_dir, exist_ok=True)
            if self.save_traj:
                self.trajectory_path = osp.join(self.work_dir, "trajectory.h5")
                self.h5_file = File(self.trajectory_path, "w")
                self.logger.info(f"Save trajectory at {self.trajectory_path}.")

        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.recent_obs = None
        self.data_episode = None
        self.video_writer = None
        self.level_index = -1
        self.logger.info(f"Begin to evaluate in worker {self.worker_id}")

        self.episode_id = -1
        self.reset()

    def done(self):
        self.episode_lens.append(self.episode_len)
        self.episode_rewards.append(self.episode_reward)
        self.episode_finishes.append(self.episode_finish)

        if self.save_traj and self.data_episode is not None:
            if (not self.only_save_success_traj) or (self.only_save_success_traj and self.episode_finish):
                group = self.h5_file.create_group(f"traj_{self.episode_id}")
                self.data_episode.to_hdf5(group, with_traj_index=False)
            self.data_episode = None
        # exit(0)
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def reset(self):
        self.episode_id += 1
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        level = None
        if self.eval_init_levels is not None:
            self.level_index = (self.level_index + 1) % len(self.eval_init_levels)
            # randomly sample a level from the self.eval_init_  levels
            # lvl = int(self.eval_init_levels[np.random.randint(len(self.eval_init_levels))])
            # print(f"Env is reset to level {lvl}")
            level = self.eval_init_levels[self.level_index]
            if isinstance(level, str):
                level = eval(level)
            self.recent_obs = self.env.reset(level=level)
        else:
            self.recent_obs = self.env.reset()
            if hasattr(self.env, "level"):
                level = self.env.level
            elif hasattr(self.env.unwrapped, "_main_seed"):
                level = self.env.unwrapped._main_seed
        if level is not None:
            extra_output = "" if self.level_index is None else f"with level id {self.level_index}"
            self.logger.info(f"Episode {self.episode_id} begins, run on level {level} {extra_output}!")

    def step(self, action):
        data_to_store = {"obs": self.recent_obs}

        if self.save_traj:
            env_state = get_env_state(self.env, self.save_scene_state)
            for key in env_state:
                data_to_store[key] = env_state[key]
            data_to_store.update(env_state)
        if self.save_video:
            image = self.env.render(mode="rgb_array")
            image = image[..., ::-1]
            if self.video_writer is None:
                self.video_file = osp.join(self.video_dir, f"{self.episode_id}.mp4")
                self.video_writer = cv2.VideoWriter(self.video_file, cv2.VideoWriter_fourcc(*"mp4v"), 20, (image.shape[1], image.shape[0]))
            self.video_writer.write(image)

        infos = self.env.step(action, to_list=False, with_info=True)

        next_obs, reward, done, info, episode_done = (
            infos["next_obs"],
            infos["rewards"][0],
            infos["dones"][0],
            infos["infos"][0],
            infos["episode_dones"][0],
        )

        self.episode_len += 1
        self.episode_reward += reward
        # print(dir(info))
        # from IPython import embed
        # embed()
        # exit(0)

        if self.log_every_step:
            info = decode_np(info)
            self.logger.info(f"Episode {self.episode_id}: Step {self.episode_len} reward: {reward}, info: {info}")
        info = str(info)

        if self.save_traj:
            data_to_store["actions"] = action
            data_to_store["next_obs"] = next_obs
            data_to_store["rewards"] = reward
            data_to_store["dones"] = done
            data_to_store["episode_dones"] = episode_done
            data_to_store["infos"] = info
            env_state = get_env_state(self.env, self.save_scene_state)
            for key in env_state:
                data_to_store[f"next_{key}"] = env_state[key]
            if self.data_episode is None:
                self.data_episode = ReplayMemory(self.horizon)
            data_to_store = GDict(data_to_store).to_numpy().f64_to_f32()
            self.data_episode.push(data_to_store)
        if episode_done:
            self.logger.info(
                f"Episode {self.episode_id} ends: Length {self.episode_len}, Reward: {self.episode_reward}, Early Stop or Finish: {done}"
            )
            self.episode_finish = done
            self.done()
            self.reset()
        else:
            self.recent_obs = next_obs
        return self.recent_obs, episode_done

    def finish(self):
        if self.save_traj:
            self.h5_file.close()

    def run(self, pi, num=1, work_dir=None, **kwargs):
        if self.eval_init_levels is not None:
            if num > len(self.eval_init_levels):
                print(f"We do not need to select more than {len(self.eval_init_levels)} levels!")
                num = min(num, len(self.eval_init_levels))

        self.start(work_dir)
        import torch
        from pyrl.utils.torch import get_cuda_info

        def reset_pi():
            if hasattr(pi, "reset"):
                assert self.worker_id is None, "Reset policy only works for single thread!"
                reset_kwargs = {}

                pi.reset(**reset_kwargs)  

        reset_pi()
        recent_obs = self.recent_obs

        while self.episode_id < num:
            with torch.no_grad():
                recent_obs = GDict(recent_obs).unsqueeze(axis=0, wrapper=False)
                action = pi(recent_obs, mode=self.sample_mode)
                action = to_np(action[0])
            # print(recent_obs, action)
            # exit(0)
            recent_obs, episode_done = self.step(action)
            if episode_done:
                reset_pi()
                print_dict = {}
                print_dict["memory"] = get_total_memory("G", False)
                print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))
                print_info = dict_to_str(print_dict)
                self.logger.info(f"{print_info}")
        self.finish()
        return self.episode_lens, self.episode_rewards, self.episode_finishes

    def close(self):
        if hasattr(self, "env"):
            del self.env
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()

    def __del__(self):
        self.close()
