python tools/run_rl.py configs/mfrl/ppo/maniskill/maniskill_pn.py --gpu-ids 0 --work-dir=./Single-Frame-Base-Movebucket --cfg-options \
'env_cfg.env_name=MoveBucket-v0' 'env_cfg.ego_mode=True' 'env_cfg.with_mask=True' \
'agent_cfg.detach_actor_feature=False' 'agent_cfg.actor_cfg.optim_cfg.lr=3e-4' 'agent_cfg.critic_cfg.optim_cfg.lr=3e-4' \
'agent_cfg.batch_size=330' 'train_rl_cfg.n_steps=40000' 'replay_cfg.capacity=40000' \
'train_rl_cfg.total_steps=15000000' \
'train_rl_cfg.n_checkpoint=2000000' 'train_rl_cfg.n_eval=15000000'