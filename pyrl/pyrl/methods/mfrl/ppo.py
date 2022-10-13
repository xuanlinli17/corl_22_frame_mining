from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pyrl.env import build_replay
from pyrl.networks import build_actor_critic, build_model
from pyrl.optimizers import build_optimizer
from pyrl.utils.data import DictArray, GDict, to_np, to_torch
from pyrl.utils.meta import get_logger
from pyrl.utils.torch import BaseAgent, RunningMeanStdTorch

from ..builder import MFRL


@MFRL.register_module()
class PPO(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        env_params,
        gamma=0.99,
        lmbda=0.95,
        max_kl=None,
        obs_norm=True,
        rew_norm=True,
        adv_norm=True,
        recompute_value=True,
        eps_clip=0.2,
        critic_coeff=0.5,
        entropy_coeff=0.0,
        num_epoch=10,
        critic_epoch=-1,
        actor_epoch=-1,
        num_mini_batch=-1,
        critic_warmup_epoch=0,
        batch_size=256,
        recurrent_horizon=-1,
        max_grad_norm=0.5,
        rms_grad_clip=None,
        dual_clip=None,
        critic_clip=False,
        shared_backbone=False,
        detach_actor_feature=False,
        debug_grad=False,
    ):
        super(PPO, self).__init__()
        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)

        actor_optim_cfg = actor_cfg.pop("optim_cfg", None)
        critic_optim_cfg = critic_cfg.pop("optim_cfg", None)
        obs_shape = env_params["obs_shape"]

        self.gamma = gamma
        self.lmbda = lmbda

        self.adv_norm = adv_norm
        self.obs_rms = RunningMeanStdTorch(obs_shape, clip_max=10) if obs_norm else None
        self.rew_rms = RunningMeanStdTorch(1) if rew_norm else None

        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.critic_clip = critic_clip
        self.max_kl = max_kl
        self.recompute_value = recompute_value
        self.max_grad_norm = max_grad_norm
        self.rms_grad_clip = rms_grad_clip
        self.debug_grad = debug_grad

        self.num_mini_batch = num_mini_batch
        self.batch_size = batch_size  # The batch size for policy gradient

        self.critic_warmup_epoch = critic_warmup_epoch
        self.num_epoch = num_epoch
        self.critic_epoch = critic_epoch
        self.actor_epoch = actor_epoch


        # Build networks
        actor_cfg.update(env_params)
        critic_cfg.update(env_params)

        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)
        self.shared_backbone = shared_backbone
        self.detach_actor_feature = detach_actor_feature

        if shared_backbone:
            self.actor_critic_optim = build_optimizer(self, actor_optim_cfg)
            self.actor_optim = self.critic_optim = self.actor_critic_optim
        else:
            self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
            self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)
        self.recurrent_horizon = recurrent_horizon

    def compute_critic_loss(self, samples):
        # For update_actor and update critic
        assert isinstance(samples, (dict, GDict))
        values, new_states = self.critic(
            samples["obs"], rnn_states=samples["rnn_states"], episode_dones=samples["episode_dones"], rnn_mode="with_states", save_feature=True
        )
        feature = self.critic.values[0].backbone.pop_attr("saved_feature")
        visual_feature = self.critic.values[0].backbone.pop_attr("saved_visual_feature")
        new_states = GDict(new_states).detach(wrapper=False)
        if new_states is not None:
            samples["new_states"] = new_states
        if self.detach_actor_feature and feature is not None:
            feature = feature.detach()

        if self.critic_clip and isinstance(self.critic_clip, float):
            v_clip = samples["old_values"] + (values - samples["old_values"]).clamp(-self.critic_clip, self.critic_clip)
            vf1 = (samples["returns"] - values).pow(2)
            vf2 = (samples["returns"] - v_clip).pow(2)
            critic_loss = torch.max(vf1, vf2)
        else:
            critic_loss = (samples["returns"] - values).pow(2)

        critic_loss = critic_loss.mean() if samples["is_valid"] is None else critic_loss[samples["is_valid"]].mean()
        return critic_loss, feature, visual_feature, new_states

    def update_actor(self, samples, with_critic=False):
        """
        Returns True if self.max_kl is not None and
        policy update causes large kl divergence between new policy and old policy,
        in which case we stop the policy update and throw away the current replay buffer
        """
        is_valid = samples["is_valid"]
        self.actor_optim.zero_grad()
        if with_critic and not self.shared_backbone:
            self.critic_optim.zero_grad()
        ret = {}
        critic_loss, actor_loss, entropy_term = [
            0,
        ] * 3

        feature, visual_feature, critic_loss, policy_std, new_states = [
            None,
        ] * 5
        if with_critic:
            critic_mse, feature, visual_feature, new_states = self.compute_critic_loss(samples)
            critic_loss = critic_mse * self.critic_coeff
            ret["ppo/critic_err"] = critic_mse.item()
            # ret['ppo/critic_loss'] = critic_loss.item()

        # Run actor forward
        tmp = self.actor(
            samples["obs"],
            rnn_states=samples["rnn_states"],
            episode_dones=samples["episode_dones"],
            mode="dist_std",
            feature=feature,
            save_feature=feature is None,
            rnn_mode="with_states" if new_states is None else "base",
            require_aux_loss=True, # auxiliary backbone self-supervision, e.g. aux_regress in VisuomotorTransformerFrame
        )
        if isinstance(tmp, dict) and 'aux_loss' in tmp.keys(): # auxiliary backbone self-supervision, e.g. aux_regress in VisuomotorTransformerFrame
            tmp, backbone_aux_loss = tmp['feat'], tmp['aux_loss']
        else:
            backbone_aux_loss = None
        if new_states is None:
            tmp, new_states = tmp
        new_distributions, policy_std = tmp
        del tmp
        if visual_feature is None:
            visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")

        # Compute actos loss
        dist_entropy = new_distributions.entropy().mean()
        recent_log_p = new_distributions.log_prob(samples["actions"])
        log_ratio = recent_log_p - samples["old_log_p"]
        ratio = log_ratio.exp()
        # print("ratio", ratio[:20], flush=True)

        # Estimation of KL divergence = p (log p - log q) with method in Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl_div = (ratio - 1 - log_ratio).mean().item()
            clip_frac = (torch.abs(ratio - 1) > self.eps_clip).float().mean().item()
            if policy_std is not None:
                ret["ppo/policy_std"] = policy_std.mean().item()
            ret["ppo/entropy"] = dist_entropy.item()
            ret["ppo/mean_p_ratio"] = ratio.mean().item()
            ret["ppo/max_p_ratio"] = ratio.max().item()
            ret["ppo/log_p"] = recent_log_p.mean().item()
            ret["ppo/clip_frac"] = clip_frac
            ret["ppo/approx_kl"] = approx_kl_div

        sign = self.max_kl is not None and approx_kl_div > self.max_kl * 1.5

        if sign:
            return True, ret

        if ratio.ndim == samples["advantages"].ndim - 1:
            ratio = ratio[..., None]

        surr1 = ratio * samples["advantages"]
        surr2 = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * samples["advantages"]
        surr = torch.min(surr1, surr2)
        if self.dual_clip:
            surr = torch.max(surr, self.dual_clip * samples["advantages"])
        actor_loss = -surr[is_valid].mean()
        entropy_term = -dist_entropy * self.entropy_coeff
        ret["ppo/actor_loss"] = actor_loss.item()
        ret["ppo/entropy_loss"] = entropy_term.item()

        loss = actor_loss + entropy_term + critic_loss
        if backbone_aux_loss is not None:
            loss = loss + backbone_aux_loss
        loss.backward()

        net = self if with_critic else self.actor
        ret["grad/grad_norm"] = net.grad_norm
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
        ret["grad/clipped_grad_norm"] = net.grad_norm

        self.actor_optim.step()
        if with_critic and not self.shared_backbone:
            self.critic_optim.step()

        return False, ret

    def update_critic(self, samples):
        self.critic_optim.zero_grad()
        critic_mse = self.compute_critic_loss(samples)[0]
        critic_loss = critic_mse * self.critic_coeff
        critic_loss.backward()

        ret = {}
        ret["grad/grad_norm"] = self.critic.grad_norm
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        ret["grad/clipped_grad_norm"] = self.critic.grad_norm
        ret["ppo/critic_loss"] = critic_loss.item()
        ret["ppo/critic_mse"] = critic_mse.item()
        self.critic_optim.step()
        return ret

    def update_parameters(self, memory, updates, with_v=False):
        logger = get_logger()
        ret = defaultdict(list)

        process_batch_size = self.batch_size if GDict(memory["obs"]).is_big else None
        if self.num_mini_batch < 0:
            max_samples = len(memory)
            num_mini_batch = int((max_samples + self.batch_size - 1) // self.batch_size)
        else:
            num_mini_batch = self.num_mini_batch
        logger.info(f"Number of batches in one PPO epoch: {num_mini_batch}!")

        if len(memory) < memory.capacity:
            memory["episode_dones"][len(memory) :] = True

        # Do transformation for all valid samples
        memory["episode_dones"] = (memory["episode_dones"] + memory["is_truncated"]) > 1 - 0.1
        if self.has_obs_processs:
            self.obs_rms.sync()
            tmp = GDict({"obs": memory["obs"], "next_obs": memory["next_obs"]}).to_torch(device="cpu", wrapper=False)
            tmp = self.process_obs(tmp, batch_size=process_batch_size)
            memory.update(tmp)
            del tmp

        with torch.no_grad():
            memory["old_distribution"], memory["old_log_p"] = self.get_dist_with_logp(
                obs=memory["obs"], actions=memory["actions"], rnn_states=memory["rnn_states"], batch_size=process_batch_size, **self.recurrent_kwargs
            )
            ret["ppo/old_log_p"].append(memory["old_log_p"].mean().item())
        def run_over_buffer(epoch_id, mode="v"):
            nonlocal memory, ret, logger
            assert mode in ["v", "pi", "v+pi"]
            if "v" in mode and (epoch_id == 0 or self.recompute_value):
                memory.update(
                    self.compute_gae(
                        obs=memory["obs"],
                        next_obs=memory["next_obs"],
                        rewards=memory["rewards"],
                        dones=memory["dones"],
                        episode_dones=memory["episode_dones"],
                        update_rms=True,
                        batch_size=process_batch_size,
                    )
                )

                if self.adv_norm:
                    # print(mean_adv, std_adv)
                    mean_adv = memory["advantages"].mean(0)
                    std_adv = memory["advantages"].std(0) + 1e-8
                    memory["advantages"] = (memory["advantages"] - mean_adv) / std_adv
                    ret["ppo/adv_mean"].append(mean_adv.item())
                    ret["ppo/adv_std"].append(std_adv.item())
                    ret["ppo/max_normed_adv"].append(np.abs(memory["advantages"]).max().item())

                ret["ppo/v_target"].append(memory["returns"].mean().item())
                ret["ppo/ori_returns"].append(memory["original_returns"].mean().item())


            def run_one_iter(samples):
                if "pi" in mode:
                    flag, infos = self.update_actor(samples, with_critic=(mode == "v+pi"))
                    for key in infos:
                        ret[key].append(infos[key])
                elif mode == "v":
                    flag, infos = False, self.update_critic(samples)
                    for key in infos:
                        ret[key].append(infos[key])
                return flag

            for samples in memory.mini_batch_sampler(self.batch_size, drop_last=True, auto_restart=True, max_num_batches=num_mini_batch):
                # print(GDict(samples).shape, GDict(samples).type)
                # exit()
                samples = DictArray(samples).to_torch(device=self.device, non_blocking=True)
                if run_one_iter(samples):
                    return True
            return False

        # logger.info(f"Begin PPO training for {self.num_epoch}!")
        # print('???')
        for i in range(self.critic_warmup_epoch):
            run_over_buffer(i, "v")
        # exit(0)
        if self.num_epoch > 0:
            for i in range(self.num_epoch):
                num_actor_epoch = i + 1
                if run_over_buffer(i, "v+pi"):
                    break
        else:
            for i in range(self.critic_epoch):
                run_over_buffer(i, "v")
            for i in range(self.actor_epoch):
                num_actor_epoch = i + 1
                if run_over_buffer(i, "pi"):
                    break
        self.critic_warmup_epoch = 0
        ret = {key: np.mean(ret[key]) for key in ret}
        ret["ppo/num_actor_epoch"] = num_actor_epoch
        return ret
