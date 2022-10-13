# norm_cfg=dict(type='LN', eps=1e-6),
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
    batch_size=330,
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
            type="VisuomotorTransformerFrame",
            visual_nn_cfg=dict(
                type="TransformerFrame",
                num_frames="nhand + 1",
                backbone_cfg=dict(
                    type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 300]
                ),
                transformer_cfg=dict(
                    type="TransformerEncoder",
                    block_cfg=dict(
                        attention_cfg=dict(
                            type="MultiHeadSelfAttention",
                            embed_dim=300,
                            num_heads=8,
                            latent_dim=32,
                            dropout=0.1,
                        ),
                        mlp_cfg=dict(
                            type="LinearMLP",
                            norm_cfg=None,
                            mlp_spec=[300, 1024, 300],
                            bias="auto",
                            inactivated_output=True,
                            linear_init_cfg=dict(type="xavier_init", gain=1, bias=0),
                        ),
                        dropout=0.1,
                    ),
                    mlp_cfg=None,
                    num_blocks=3,
                ),
                mask_type="full",
            ),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=['300 + agent_shape', 192, 128],
                inactivated_output=True,
                zero_init_output=True,
            ),
            is_value=False,
        ),
        optim_cfg=dict(type="Adam", lr=1e-4), # 3e-4 leads to unstable training
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        nn_cfg=dict(
            type="VisuomotorTransformerFrame",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP", norm_cfg=None, mlp_spec=['300 + agent_shape', 192, 128, 1], inactivated_output=True, zero_init_output=True
            ),
            is_value=True,
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
    nhand_pose=1,
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
    total_steps=int(1e8),
    n_steps=40000,
    n_eval=int(5e6),
    n_checkpoint=int(1e6),
)

eval_cfg = dict(
    type="Evaluation",
    num=100,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=True,
    debug_print=False,
    env_cfg=dict(no_early_stop=False),
)