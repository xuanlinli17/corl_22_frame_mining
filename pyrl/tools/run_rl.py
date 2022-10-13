import argparse
import glob
import os
import os.path as osp
import shutil
import time
import warnings
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np

warnings.simplefilter(action="ignore")


from pyrl.utils.data import is_not_null, is_null, num_to_str
from pyrl.utils.meta import (
    Config,
    DictAction,
    add_env_var,
    get_logger,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL training code")
    # Configurations
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--cfg-options",
        "--opt",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    # Parameters for log dir
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--dev", action="store_true", default=False, help="add timestamp to the name of work dir")
    parser.add_argument("--with-agent-type", default=False, action="store_true", help="add agent type to work dir")
    parser.add_argument(
        "--agent-type-first",
        default=False,
        action="store_true",
        help="when work-dir is None, we will use agent_type/config_name or config_name/agent_type",
    )
    parser.add_argument("--clean-up", help="Clean up the work_dir", action="store_true")

    # If we use evaluation mode
    parser.add_argument("--evaluation", "--eval", help="use evaluation mode", action="store_true")

    # If we resume checkpoint model
    parser.add_argument("--resume-from", default=None, nargs="+", help="the checkpoint file to resume from")
    parser.add_argument(
        "--auto-resume", help="auto-resume the checkpoint under work dir, " "the default value is true when in evaluation mode", action="store_true"
    )
    parser.add_argument("--resume-keys-map", default=None, nargs="+", action=DictAction, help="specify how to change the keys in checkpoints")

    # Specify GPU
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--num-gpus", default=None, type=int, help="number of gpus to use")
    group_gpus.add_argument("--gpu-ids", default=None, type=int, nargs="+", help="ids of gpus to use")
    group_gpus.add_argument("--sim-gpu-ids", default=None, type=int, nargs="+", help="ids of gpus to do simulation")

    # Torch and reproducibility settings
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="whether to use benchmark mode in cudnn.")

    args = parser.parse_args()

    # Merge cfg with args.cfg_options
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        for key, value in args.cfg_options.items():
            try:
                value = eval(value)
                args.cfg_options[key] = value
            except:
                pass
        cfg.merge_from_dict(args.cfg_options)

    args.with_agent_type = args.with_agent_type or args.agent_type_first
    for key in ["work_dir", "env_cfg", "resume_from", "eval_cfg", "replay_cfg", "rollout_cfg"]:
        cfg[key] = cfg.get(key, None)
    if args.seed is None:
        args.seed = np.random.randint(2**32 - 10000)
    args.mode = "eval" if args.evaluation else "train"
    return args, cfg


def build_work_dir():
    if is_null(args.work_dir):
        root_dir = "./work_dirs"
        env_name = cfg.env_cfg.get("env_name", None) if is_not_null(cfg.env_cfg) else None
        config_name = osp.splitext(osp.basename(args.config))[0]
        folder_name = env_name if is_not_null(env_name) else config_name
        if args.with_agent_type:
            if args.agent_type_first:
                args.work_dir = osp.join(root_dir, agent_type, folder_name)
            else:
                args.work_dir = osp.join(root_dir, folder_name, agent_type)
        else:
            args.work_dir = osp.join(root_dir, folder_name)
    elif args.with_agent_type:
        if args.agent_type_first:
            print("When you specify the work dir path, the agent type cannot be first!", level="warning")
        args.work_dir = osp.join(args.work_dir, agent_type)

    if args.dev:
        args.work_dir = osp.join(args.work_dir, args.timestamp)

    if args.clean_up:
        if args.evaluation or args.auto_resume or (is_not_null(args.resume_from) and os.path.commonprefix(args.resume_from) == args.work_dir):
            print("We will ignore the clean-up flag, when we are in evaluation mode or resume from the directory!", level="warning")
        else:
            shutil.rmtree(args.work_dir, ignore_errors=True)
    os.makedirs(osp.abspath(args.work_dir), exist_ok=True)


def find_checkpoint():
    logger = get_logger()
    if is_not_null(args.resume_from):
        if is_not_null(cfg.resume_from):
            print(f"The resumed checkpoint in config file is overwrited by {args.resume_from}!", level="warning")
        cfg.resume_from = args.resume_from

    if args.auto_resume or (args.evaluation and is_null(cfg.resume_from)):
        logger.info(f"Search model under {args.work_dir}.")
        model_names = list(glob.glob(osp.join(args.work_dir, "models", "*.ckpt")))
        latest_index = -1
        latest_name = None
        for model_i in model_names:
            index_str = osp.basename(model_i).split(".")[0].split("_")[1]
            if index_str == "final":
                latest_name = model_i
                break
            index = eval(index_str)
            if index > latest_index:
                latest_index = index
                latest_name = model_i

        if is_null(latest_name):
            print(f"Find no checkpoints under {args.work_dir}!")
        else:
            cfg.resume_from = latest_name
    if is_not_null(cfg.resume_from):
        if isinstance(cfg.resume_from, str):
            cfg.resume_from = [
                cfg.resume_from,
            ]
        logger.info(f"Get {len(cfg.resume_from)} checkpoint {cfg.resume_from}.")
        logger.info(f"Check checkpoint {cfg.resume_from}!")

        for file in cfg.resume_from:
            if not (osp.exists(file) and osp.isfile(file)):
                logger.error(f"Checkpoint file {file} does not exist!")
                exit(-1)

def init_torch(args):
    import torch

    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.gpu_ids is not None and len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
        torch.set_num_threads(1)

    torch.manual_seed(args.seed + 0)
    torch.cuda.manual_seed_all(args.seed + 0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_rl(rollout, evaluator, replay, args, cfg):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from pyrl.apis.train_rl import train_rl
    from pyrl.env import save_eval_statistics
    from pyrl.methods.builder import build_agent
    from pyrl.utils.data.converter import dict_to_str
    from pyrl.utils.torch import BaseAgent, load_checkpoint, save_checkpoint

    logger = get_logger()
    logger.info("Initialize torch!")
    init_torch(args)
    logger.info("Finish Initialize torch!")

    if is_not_null(cfg.agent_cfg.get("batch_size", None)) and isinstance(cfg.agent_cfg.batch_size, (list, tuple)):
        assert len(cfg.agent_cfg.batch_size) == len(args.gpu_ids)
        cfg.agent_cfg.batch_size = cfg.agent_cfg.batch_size[0]
        logger.info(f"Set batch size to {cfg.agent_cfg.batch_size}!")

    logger.info("Build agent!")
    agent = build_agent(cfg.agent_cfg)
    assert agent is not None, f"Agent type {cfg.agent_cfg.type} is not valid!"

    logger.info(agent)
    logger.info(
        f'Num of parameters: {num_to_str(agent.num_trainable_parameters, unit="M")}, Model Size: {num_to_str(agent.size_trainable_parameters, unit="M")}'
    )
    device = "cpu" if len(args.gpu_ids) == 0 else "cuda"
    agent = agent.float().to(device)
    assert isinstance(agent, BaseAgent), "The agent object should be an instance of Agent!"

    if is_not_null(cfg.resume_from):
        logger.info("Resume agent with checkpoint!")
        for file in cfg.resume_from:
            load_checkpoint(agent, file, device, keys_map=args.resume_keys_map, logger=logger)

    logger.info(f"Work directory of this run {args.work_dir}")
    if len(args.gpu_ids) > 0:
        logger.info(f"Train over GPU {args.gpu_ids}!")
    else:
        logger.info(f"Train over CPU!")

    if not args.evaluation:
        train_rl(
            agent,
            rollout,
            evaluator,
            replay,
            work_dir=args.work_dir,
            eval_cfg=cfg.eval_cfg,
            **cfg.train_rl_cfg,
        )
    else:
        agent.eval()
        agent.set_mode("test")

        if is_not_null(evaluator):
            # For RL
            lens, rewards, finishes = evaluator.run(agent, work_dir=work_dir, **cfg.eval_cfg)
            save_eval_statistics(work_dir, lens, rewards, finishes)
        agent.train()
        agent.set_mode("train")

    if len(args.gpu_ids) > 1:
        dist.destroy_process_group()


def run_one_process(rank, world_size, args, cfg):
    import numpy as np

    np.set_printoptions(3)

    set_random_seed(args.seed + rank)

    if is_not_null(cfg.env_cfg) and len(args.gpu_ids) > 0:
        if args.sim_gpu_ids is not None:
            assert len(args.sim_gpu_ids) == len(args.gpu_ids), "Number of simulation gpus should be the same as the training gpus recently!"
        else:
            args.sim_gpu_ids = args.gpu_ids
        cfg.env_cfg.device = f"cuda:{args.sim_gpu_ids[rank]}"

    work_dir = args.work_dir
    logger_file = osp.join(work_dir, f"{args.timestamp}-{args.name_suffix}.log")
    logger = get_logger(name=None, log_file=logger_file, log_level=cfg.get("log_level", "INFO"))

    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Set random seed to {args.seed}")

    if is_not_null(cfg.replay_cfg) and (not args.evaluation):
        logger.info(f"Build replay buffer!")
        from pyrl.env import build_replay

        replay = build_replay(cfg.replay_cfg)
    else:
        replay = None

    if not args.evaluation and is_not_null(cfg.rollout_cfg):
        from pyrl.env import build_rollout

        logger.info(f"Build rollout!")
        rollout_cfg = cfg.rollout_cfg
        rollout_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
        rollout = build_rollout(rollout_cfg)
    else:
        rollout = None

    if is_not_null(cfg.eval_cfg) and rank == 0:
        # Only the first process will do evaluation
        from pyrl.env import build_evaluation

        logger.info(f"Build evaluation!")
        eval_cfg = cfg.eval_cfg
        # Evaluation environment setup can be different from the training set-up. (Like eraly-stop or object sets)
        if eval_cfg.get("env_cfg", None) is None:
            eval_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
        else:
            if eval_cfg["env_cfg"] is not None:
                tmp = eval_cfg["env_cfg"]
                eval_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
                eval_cfg["env_cfg"].update(tmp)
        get_logger().info(f"Building evaluation: eval_cfg: {eval_cfg}")
        evaluator = build_evaluation(eval_cfg)
    else:
        evaluator = None
    obs_shape, action_shape = None, None
    if is_not_null(cfg.env_cfg):
        # For RL which needs environments
        logger.info(f"Get obs shape!")
        from pyrl.env import get_env_info

        env_params = get_env_info(cfg.env_cfg)
        cfg.agent_cfg["env_params"] = env_params
        obs_shape = env_params["obs_shape"]
        action_shape = env_params["action_shape"]
        logger.info(f'State shape:{env_params["obs_shape"]}, action shape:{env_params["action_shape"]}')
    elif is_not_null(replay):
        obs_shape = None
        for obs_key in ["inputs", "obs"]:
            if obs_key in replay.memory:
                obs_shape = replay.memory.take(0).shape.memory[obs_key]
                break

    if is_not_null(obs_shape) or is_not_null(action_shape):
        from pyrl.networks.utils import get_kwargs_from_shape, replace_placeholder_with_args

        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)

    main_rl(rollout, evaluator, replay, args, cfg)

    if is_not_null(evaluator):
        evaluator.close()
        logger.info("Close evaluator object")
    if is_not_null(rollout):
        rollout.close()
        logger.info("Close rollout object")
    if is_not_null(replay):
        del replay
        logger.info("Delete replay buffer")
    # flush_logger(logger)


def main():
    if len(args.gpu_ids) > 1:
        import torch.multiprocessing as mp

        world_size = len(args.gpu_ids)
        mp.spawn(run_one_process, args=(world_size, args, cfg), nprocs=world_size, join=True)
    else:
        run_one_process(0, 1, args, cfg)


if __name__ == "__main__":
    add_env_var()

    args, cfg = parse_args()
    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    agent_type = cfg.agent_cfg.type

    build_work_dir()
    find_checkpoint()

    work_dir = args.work_dir
    if args.evaluation:
        test_name = "test"
        work_dir = osp.join(work_dir, test_name)
        # Always clean up for evaluation
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir

    logger_name = cfg.env_cfg.env_name if is_not_null(cfg.env_cfg) else cfg.agent_cfg.type
    args.name_suffix = f"{args.mode}"
    os.environ["PYRL_LOGGER_NAME"] = f"{logger_name}-{args.name_suffix}"
    cfg.dump(osp.join(work_dir, f"{args.timestamp}-{args.name_suffix}.py"))

    main()
