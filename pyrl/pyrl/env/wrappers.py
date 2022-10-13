from collections import deque

import cv2
import numpy as np
from gym import spaces
from gym.core import ActionWrapper, ObservationWrapper, Wrapper
from gym.spaces import Discrete
from pyrl.utils.data import DictArray, GDict, deepcopy, encode_np, is_num, to_array
from pyrl.utils.meta import Registry, build_from_cfg

from .observation_process import (
    pcd_base,
    pcd_uniform_downsample,
)

WRAPPERS = Registry("wrapper of env")


class TrivialRolloutWrapper(Wrapper):
    """
    The same api as VectorEnv
    """

    def __init__(self, env, use_cost=False, reward_scale=1, extra_dim=False):
        super(TrivialRolloutWrapper, self).__init__(env)
        self.recent_obs = None
        self.iscost = -1 if use_cost else 1
        self.num_envs = 1
        self.reward_scale = reward_scale
        self.is_discrete = isinstance(env.action_space, Discrete)
        self.extra_dim = extra_dim
        self.episode_dones = False
        self.ready_idx = []
        self.running_idx = []
        self.idle_idx = 0

    @property
    def done_idx(self):
        if self.episode_dones:
            return [
                0,
            ]
        else:
            return []

    def act_process(self, action):
        if self.is_discrete:
            if is_num(action):
                action = int(action)
            else:
                action = action.reshape(-1)
                assert len(action) == 1, f"Dim of discrete action should be 1, but we get {len(action)}"
                action = int(action[0])
        return action

    # For original gym interface.
    def random_action(self):
        action = to_array(self.action_space.sample())
        if self.extra_dim:
            action = action[None]
        return action

    def reset(self, *args, idx=None, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        obs = GDict(obs).f64_to_f32(False)
        if self.extra_dim:
            obs = GDict(obs).unsqueeze(0, False)
        obs = deepcopy(obs)
        self.recent_obs = obs
        return obs

    def reset_dict(self, idx=None, **kwargs):
        return {"obs": self.reset(**kwargs)}

    def render_dict(self, idx=None, **kwargs):
        return {"images": self.render(**kwargs)}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        else:
            return getattr(self.env, name)

    def step_dict(self, action, with_info=False, encode_info=True, **kwargs):
        """
        Change the output of step to a dict format !!
        """
        from .env_utils import true_done

        obs = self.recent_obs
        if self.extra_dim:
            obs = GDict(obs).take(0, 0, False)
            if action.ndim > 1:
                action = action[0]
        next_obs, rewards, episode_dones, infos = self.env.step(self.act_process(action))
        next_obs = GDict(next_obs).f64_to_f32(False)
        next_obs = deepcopy(next_obs)
        # episode_dones = episode_dones
        # rewards = deepcopy(rewards)
        self.episode_dones = episode_dones
        infos = deepcopy(infos)

        dones = true_done(episode_dones, infos)
        dones = np.array([dones], dtype=np.bool)
        rewards = np.array([rewards * self.reward_scale], dtype=np.float32)
        episode_dones = np.array([episode_dones], dtype=np.bool)
        ret = {
            "obs": obs,
            "actions": action,
            "next_obs": next_obs,
            "rewards": rewards,
            "dones": dones,
            "episode_dones": episode_dones,
        }
        if with_info:
            if "TimeLimit.truncated" not in infos:
                infos["TimeLimit.truncated"] = False
            ret["infos"] = (
                np.array([encode_np(infos)], dtype=np.dtype("U10000")) if encode_info else GDict(infos).to_array(wrapper=False)
            )  # info with at most 10KB
        if self.extra_dim:
            ret = GDict(ret).unsqueeze(0, False)
        # print(GDict(ret).shape)

        self.recent_obs = ret["next_obs"]
        return ret

    def step(self, action, with_info=False, to_list=True, encode_info=True, **kwargs):
        infos = self.step_dict(action, with_info, encode_info)
        return (infos["next_obs"], infos["rewards"][0], infos["episode_dones"][0], infos.get("infos", {})) if to_list else infos

    def step_random_actions(self, num, with_info=False, encode_info=True):
        ret = []
        for i in range(num):
            obs = self.recent_obs
            item = self.step_dict(self.random_action(), encode_info=encode_info, with_info=with_info)
            item["obs"] = obs
            ret.append(item)
            if item["episode_dones"]:
                self.reset()
        if self.extra_dim:
            ret = DictArray.concat(ret, axis=0, wrapper=False)
        else:
            ret = DictArray.stack(ret, axis=0, wrapper=False)
        return ret

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps


class ManiSkillObsWrapper(ObservationWrapper):
    def __init__(self, env, process_mode="base", n_points=1200, ret_orig_pcd=False, nhand_pose=0):
        """
        Stack k last frames for point clouds or rgbd and remap the rendering configs
        """
        super(ManiSkillObsWrapper, self).__init__(env)
        self.process_mode = process_mode
        self.n_points = n_points # if point cloud obs, num of points in a point cloud to keep
        self.ret_orig_pcd = ret_orig_pcd # if point cloud obs, whether to return original (non-downsampled) points with ground removed
        self.nhand_pose = nhand_pose # if nhand_pose > 0, return the poses of the nhand_pose end-effectors

    def get_state(self):
        return self.env.get_state(True)

    def observation(self, observation):
        if self.obs_mode == "state":
            return np.concatenate([observation])

        if self.ret_orig_pcd:
            import copy
            obs_orig = pcd_uniform_downsample(copy.deepcopy(observation), self.env, num=10000)
            pcd_orig = obs_orig[self.obs_mode]
            inst_seg_orig = obs_orig["inst_seg"]
        if self.process_mode == "base":
            observation = pcd_base(observation, self.env, num=self.n_points)
        elif self.process_mode is not None:
            print(self.process_mode)
            raise ValueError
        visual_data = observation[self.obs_mode]

        state = observation["state"] if "state" in observation else observation["agent"]
        if self.nhand_pose > 0:
            # extract the hand pose information from the state vector
            hp_dim = self.nhand_pose * 7
            state, hand_pose = state[:-hp_dim], state[-hp_dim:]
        if "target_info" in observation: # ego-centric push target in PushChair
            target_info = observation.pop("target_info")
            state = np.concatenate([target_info, state])

        # Convert dict of array to list of array with sorted key
        ret = {}
        ret[self.obs_mode] = visual_data
        if self.ret_orig_pcd:
            ret[f"orig_{self.obs_mode}"] = pcd_orig
            ret["orig_inst_seg"] = inst_seg_orig
        ret["state"] = state
        for key in observation:
            if key not in [self.obs_mode, "state", "agent"]:
                ret[key] = observation[key]
        if self.nhand_pose > 0:
            ret["hand_pose"] = np.reshape(hand_pose, [self.nhand_pose, 7])
        return ret

    def step(self, action):
        next_obs, reward, done, info = super(ManiSkillObsWrapper, self).step(action)
        return next_obs, reward, done, info

    def reset(self, level=None):
        return self.observation(self.env.reset() if level is None else self.env.reset(level=level))

    def get_obs(self):
        return self.observation(self.env.get_obs())

    def set_state(self, *args, **kwargs):
        return self.observation(self.env.set_state(*args, **kwargs))

    def render(self, mode="human", *args, **kwargs):
        if mode == "human":
            self.env.render(mode, *args, **kwargs)
            return

        if mode in ["rgb_array", "color_image"]:
            img = self.env.render(mode="color_image", *args, **kwargs)
        else:
            img = self.env.render(mode=mode, *args, **kwargs)
        if isinstance(img, dict):
            if "world" in img:
                img = img["world"]
            elif "main" in img:
                img = img["main"]
            else:
                print(img.keys())
                exit(0)
        if isinstance(img, dict):
            img = img["rgb"]
        if img.ndim == 4:
            assert img.shape[0] == 1
            img = img[0]
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, a_min=0, a_max=1) * 255
        img = img[..., :3]
        img = img.astype(np.uint8)
        return img


def build_wrapper(cfg, default_args=None):
    return build_from_cfg(cfg, WRAPPERS, default_args)
