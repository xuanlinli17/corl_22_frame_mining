agent_cfg = dict(
    type="PPO",
    gamma=0.95,
    lmbda=0.95,
    critic_coeff=1,
    entropy_coeff=0,
    critic_clip=False,
    obs_norm=False,
    rew_norm=True,
    adv_norm=True,
    recompute_value=True,
    num_epoch=2,
    critic_warmup_epoch=4,
    batch_size=400,
    detach_actor_feature=False,
    max_grad_norm=0.5,
    eps_clip=0.2,
    max_kl=0.2,
    dual_clip=None,
    shared_backbone=True,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="GaussianHead",
            init_log_std=-1,
            clip_return=True,
            predict_std=False,
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="SparseResNet", in_channel="pcd_all_channel", voxel_size=0.05, depth=10, dropout=0.0),
            mlp_cfg=dict(type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape", 256, 256, "action_shape"], inactivated_output=True),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="SparseResNet", in_channel="pcd_all_channel", voxel_size=0.05, depth=10, dropout=0.0),
            mlp_cfg=dict(type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape", 256, 256, 1], inactivated_output=True),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
)


env_cfg = dict(
    type="gym",
    env_name="OpenCabinetDoor-v0",
    unwrapped=False,
    obs_mode="pointcloud",
    with_ext_torque=True,
    no_early_stop=True,
    cos_sin_representation=True,
    reward_scale=0.3,
    ego_mode=True,
    with_mask=True,
    process_mode="base",
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=5,
    sync=True,
    shared_memory=True,
    with_info=False,
)

replay_cfg = dict(
    type="ReplayMemory",
    capacity=40000,
    sampling_cfg=dict(type="OneStepTransition", with_replacement=False),
)


train_rl_cfg = dict(
    on_policy=True,
    warm_steps=0,
    total_steps=int(2e7),
    n_steps=40000,
    n_eval=int(5e6),
    n_checkpoint=int(1e6),
)

eval_cfg = dict(
    type="Evaluation",
    num=100,
    num_procs=5,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=True,
    debug_print=False,
    env_cfg=dict(no_early_stop=False),
)
