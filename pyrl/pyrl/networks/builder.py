import torch.nn as nn

from ..utils.meta import Registry, build_from_cfg

BACKBONES = Registry("backbone")
APPLICATIONS = Registry("applications")
REGHEADS = Registry("regression_head")

POLICYNETWORKS = Registry("policy_network")
VALUENETWORKS = Registry("value_network")


def build(cfg, registry, default_args=None):
    if cfg is None:
        return None
    elif isinstance(cfg, (list, tuple)):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return modules  # nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_reg_head(cfg):
    return build(cfg, REGHEADS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_model(cfg, default_args=None):
    if cfg is None:
        return None
    for model_type in [BACKBONES, POLICYNETWORKS, VALUENETWORKS]:
        if cfg["type"] in model_type.module_dict:
            return build(cfg, model_type, default_args)
    raise RuntimeError(f"No this model type:{cfg['type']}!")


def build_actor_critic(actor_cfg, critic_cfg, shared_backbone=False):
    if shared_backbone:
        assert "Visuomotor" in actor_cfg.nn_cfg.type or actor_cfg.nn_cfg.type in ["RNN"], (
            "Only Visuomotor model could share backbone, actually visual backbone can be shared.")
        # ensure consistency of actor_cfg and critic_cfg regarding the use of canonical object frames
        if hasattr(actor_cfg.nn_cfg, 'use_obj_frames') or hasattr(critic_cfg.nn_cfg, 'use_obj_frames'):
            assert (hasattr(actor_cfg.nn_cfg, 'use_obj_frames') and hasattr(critic_cfg.nn_cfg, 'use_obj_frames')
                and actor_cfg.nn_cfg.use_obj_frames == critic_cfg.nn_cfg.use_obj_frames), \
                "The use of object frame should be consistent between actor and critic"
        if hasattr(actor_cfg.nn_cfg, 'obj_frame_rot') or hasattr(critic_cfg.nn_cfg, 'obj_frame_rot'):
            assert (hasattr(actor_cfg.nn_cfg, 'obj_frame_rot') and hasattr(critic_cfg.nn_cfg, 'obj_frame_rot')
                and actor_cfg.nn_cfg.obj_frame_rot == critic_cfg.nn_cfg.obj_frame_rot), \
                "The use of object frame should be consistent between actor and critic"      

        actor = build_model(actor_cfg)

        if getattr(actor.backbone, "visual_nn", None) is not None:
            critic_cfg.nn_cfg.visual_nn_cfg = None
            critic_cfg.nn_cfg.visual_nn = actor.backbone.visual_nn

        critic = build_model(critic_cfg)
        return actor, critic
    else:
        return build_model(actor_cfg), build_model(critic_cfg)
