from copy import deepcopy
from pathlib import Path

import gym
from gym.envs import registry
from gym.spaces import Box, Discrete
from gym.wrappers import TimeLimit
from pyrl.utils.data import GDict
from pyrl.utils.meta import Registry, build_from_cfg, get_logger
from .wrappers import ManiSkillObsWrapper, TrivialRolloutWrapper, build_wrapper

ENVS = Registry("env")
_REG = {}


def import_env():
    try:
        _REG["sapien"] = 1
        import contextlib
        import os

        os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
        with contextlib.redirect_stdout(None):
            import mani_skill.env
    except ImportError:
        pass

def get_gym_env_type(env_name):
    import_env()
    if env_name not in registry.env_specs:
        raise ValueError("No such env")
    entry_point = registry.env_specs[env_name].entry_point
    if entry_point.startswith("gym.envs."):
        type_name = entry_point[len("gym.envs.") :].split(":")[0].split(".")[0]
    else:
        type_name = entry_point.split(".")[0]
    return type_name


def get_env_state(env, save_scene_state):
    ret = {}
    if hasattr(env, "get_state"):
        ret["env_states"] = env.get_state()
    if hasattr(env.unwrapped, "_scene") and save_scene_state:
        ret["env_scene_states"] = env.unwrapped._scene.pack()
    if hasattr(env, "level"):
        ret["env_levels"] = env.level
    return ret


def true_done(done, info):
    # Process gym standard time limit wrapper
    if done:
        if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
            return False
        else:
            return True
    return False


def get_env_info(env_cfg):
    """
    For observation space, we use obs_shape instead of gym observation space which is not easy to use in network building!
    """
    env_cfg = env_cfg.copy()
    env = build_single_env(env_cfg)
    obs = env.reset()
    obs_shape = GDict(obs).shape.memory
    action = env.random_action()
    action_space = deepcopy(env.action_space)
    action_shape = action.shape[-1]
    get_logger().info(f"Environment has the continuous action space with dimension {action_shape}.")
    del env
    return dict(obs_shape=obs_shape, action_shape=action_shape, action_space=action_space)


def get_max_episode_steps(env):
    if isinstance(env, TimeLimit):
        return env._max_episode_steps
    elif hasattr(env.unwrapped, "_max_episode_steps"):
        # For env does not use TimeLimit, e.g. ManiSkill
        return env.unwrapped._max_episode_steps
    else:
        return None


def make_gym_env(
    env_name, unwrapped=False, horizon=None, time_horizon_factor=1, stack_frame=1, 
    use_cost=False, reward_scale=1, extra_dim=False, **kwargs
):
    """
    If we want to add custom wrapper, we need to unwrap the env if the original env is wrapped by TimeLimit wrapper.
    All environments will have TrivialRolloutWrapper outside.
    """

    import_env()
    kwargs = deepcopy(kwargs)

    env_type = get_gym_env_type(env_name)
    if env_type not in [
        "mani_skill",
        "mani_skill2",
    ]:
        # For environments that cannot specify GPU, we pop device
        kwargs.pop("device", None)

    extra_wrappers = kwargs.pop("extra_wrappers", None)
    if env_type == "mani_skill":
        # Extra kwargs for maniskill1
        remove_visual = kwargs.pop("remove_visual", False)
        process_mode = kwargs.pop("process_mode", "base")
        ret_orig_pcd = kwargs.pop("ret_orig_pcd", False)
        n_points = kwargs.pop("n_points", 1200)
        nhand_pose = kwargs.pop("nhand_pose", 0) # if > 0, return the pose of nhand_pose hands in the state vector
        if nhand_pose > 0:
            kwargs['with_hand_pose'] = True
            if "OpenCabinet" in env_name:
                assert nhand_pose == 1, "OpenCabinetDrawer/Door are single arm envs"
            elif "Chair" in env_name or "Bucket" in env_name:
                assert nhand_pose == 2, "PushChair/MoveBucket are dual arm envs"
            else:
                assert NotImplementedError

    env = gym.make(env_name, **kwargs)

    if env is None:
        print(f"No {env_name} in gym")
        exit(0)

    use_time_limit = False
    max_episode_steps = get_max_episode_steps(env) if horizon is None else int(horizon)
    if isinstance(env, TimeLimit):
        env = env.env
        use_time_limit = True
    elif hasattr(env.unwrapped, "_max_episode_steps"):
        if horizon is not None:
            env.unwrapped._max_episode_steps = int(horizon)
        else:
            env.unwrapped._max_episode_steps = int(max_episode_steps * time_horizon_factor)

    if unwrapped:
        env = env.unwrapped if hasattr(env, "unwrapped") else env

    if env_type == "mani_skill":
        env = ManiSkillObsWrapper(env, process_mode, n_points=n_points, ret_orig_pcd=ret_orig_pcd, nhand_pose=nhand_pose)
    else:
        print(f"Unsupported env_type({env_type}) of {env_name}")

    if extra_wrappers is not None:
        if not isinstance(extra_wrappers, list):
            extra_wrappers = [
                extra_wrappers,
            ]
        for extra_wrapper in extra_wrappers:
            extra_wrapper.env = env
            extra_wrapper.env_name = env_name
            env = build_wrapper(extra_wrapper)
    if use_time_limit and not unwrapped:
        env = TimeLimit(env, int(max_episode_steps * time_horizon_factor))
    env = TrivialRolloutWrapper(
        env, use_cost, reward_scale, extra_dim=extra_dim
    )  # Speed up some special cases like random exploration with vectorized env
    return env


ENVS.register_module("gym", module=make_gym_env)


def build_single_env(cfg, extra_dim=False, worker_id=None):
    import_env()
    cfg["extra_dim"] = extra_dim
    return build_from_cfg(cfg, ENVS)


def build_env(cfgs, num_procs=None, single_procs=True, **vec_env_kwargs):
    # add a wrapper to gym.make to make the environment building process more flexible.
    import_env()

    num_procs = num_procs or 1
    if isinstance(cfgs, dict):
        cfgs = [cfgs] * num_procs
    if len(cfgs) == 1 and single_procs:
        return build_single_env(cfgs[0], extra_dim=True)
    else:
        from .vec_env import VectorEnv

        return VectorEnv(cfgs, **vec_env_kwargs)
