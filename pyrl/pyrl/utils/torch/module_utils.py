from torch.nn import Module, ModuleList, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parameters_to_vector
from contextlib import contextmanager
import torch, numpy as np
import math

from pyrl.utils.data import GDict, DictArray, to_torch
from pyrl.utils.meta import get_logger
from .misc import no_grad, mini_batch, run_with_mini_batch


class ExtendedModuleBase(Module):
    def __init__(self, *args, **kwargs):
        super(ExtendedModuleBase, self).__init__(*args, **kwargs)
        self._in_test = False  # For RL test mode ( do not update obs_rms and rew_rms )
        self.is_recurrent = False

    def set_mode(self, mode="train"):
        self._in_test = mode == "test"
        for module in self.children():
            if isinstance(module, ExtendedModuleBase):
                module.set_mode(mode)
        return self

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def trainable_parameters(self):
        return [_ for _ in self.parameters() if _.requires_grad]

    @property
    def size_trainable_parameters(self):
        return GDict([_ for _ in self.parameters() if _.requires_grad]).nbytes_all

    @property
    def num_trainable_parameters(self):
        return sum([_.numel() for _ in self.parameters() if _.requires_grad])

    @property
    @no_grad
    def grad_norm(self, ord=2):
        grads = [torch.norm(_.grad.detach(), ord) for _ in self.parameters() if _.requires_grad and _.grad is not None]
        ret = torch.norm(torch.stack(grads), ord).item() if len(grads) > 0 else 0.0
        if math.isnan(ret):
            print(
                """
                WARNING:
                ------------------------------------------------------
                Gradient is nan!
                ------------------------------------------------------
                """
                , flush=True
            )
            print(
                """
                ------------------------------------------------------
                Nan Parameter info:
                ------------------------------------------------------
                """
                , flush=True
            )
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None and math.isnan(torch.norm(param.grad.detach(), ord).item()):
                    print(name, param, param.grad.detach(), flush=True)            
            print(
                """
                ------------------------------------------------------
                Full Parameter info:
                ------------------------------------------------------
                """
                , flush=True
            )
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(name, param, param.grad.detach(), flush=True)
        return ret

    @no_grad
    def vector_parameters(self):
        return parameters_to_vector(self.parameters())

    def pop_attr(self, name):
        if hasattr(self, name):
            ret = getattr(self, name)
            setattr(self, name, None)
            return ret
        else:
            return None


class ExtendedModule(ExtendedModuleBase):
    # DDP has attribute device!!!!
    @property
    def device(self):
        return next(self.parameters()).device


class ExtendedModuleList(ModuleList, ExtendedModule):
    @property
    def device(self):
        return next(self.parameters()).device


class ExtendedSequential(Sequential, ExtendedModule):
    def append(self, module):
        index = len(self)
        self.add_module(str(index), module)

    def append_list(self, modules):
        assert isinstance(modules, Sequential) or isinstance(modules, (list, tuple))
        for module in modules:
            self.append(module)


class BaseAgent(ExtendedModule):
    def __init__(self, *args, **kwargs):
        super(BaseAgent, self).__init__(*args, **kwargs)
        self._device_ids = None
        self._be_data_parallel = False
        self._tmp_attrs = {}

        self.obs_processor = None
        self.obs_rms = None
        self.rew_rms = None
        self.batch_size = None
        self.recurrent_horizon = -1

    def reset(self, *args, **kwargs):
        pass

    @property
    def recurrent_kwargs(self):
        return dict(is_recurrent=self.is_recurrent, recurrent_horizon=self.recurrent_horizon)

    @property
    def has_obs_processs(self):
        return self.obs_rms is not None or self.obs_processor is not None

    @no_grad
    def process_obs(self, data, **kwargs):
        for key in ["obs", "next_obs"]:
            if key in data:
                if self.obs_rms is not None:
                    data[key] = run_with_mini_batch(self.obs_rms.normalize, data[key], **kwargs, device=self.device, wrapper=False)
                if self.obs_processor is not None:
                    data[key] = run_with_mini_batch(self.obs_processor, {"obs": data[key]}, **kwargs)["obs"]
        return data

    @no_grad
    def forward(self, obs, **kwargs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs) if self._in_test else self.obs_rms.add(obs)
        if self.obs_processor is not None:
            obs = self.obs_processor({"obs": obs})["obs"]
        return self.actor(obs, **kwargs)

    def get_dist_with_logp(self, obs, actions=None, **kwargs):
        @mini_batch(False)
        def run(obs, actions, **kwargs):
            ret = self.actor(obs, mode="dist", **kwargs)
            if actions is not None:
                if isinstance(ret, (list, tuple)):
                    ret = [ret[0], ret[0].log_prob(actions)], ret[1]
                else:
                    ret = [ret, ret.log_prob(actions)]
            return ret

        return run(obs=obs, actions=actions, **kwargs, device=self.device)

    def get_values(self, obs, actions=None, **kwargs):
        return run_with_mini_batch(self.critic, obs=obs, actions=actions, **kwargs, device=self.device)

    @no_grad
    def compute_gae(self, obs, next_obs, rewards, dones, episode_dones, update_rms=True, batch_size=None):
        """
        High-Dimensional Continuous Control Using Generalized Advantage Estimation
            https://arxiv.org/abs/1506.02438
        """
        rewards = to_torch(rewards, device=self.device, non_blocking=True)
        dones = to_torch(dones, device=self.device, non_blocking=True)
        episode_dones = to_torch(episode_dones, device=self.device, non_blocking=True)
        episode_masks = 1.0 - episode_dones.float()
        # get_logger().info(type(episode_dones))
        values, [rnn_states, rnn_next_states, rnn_final_states] = self.get_values(
            obs=obs,
            batch_size=batch_size,
            ret_device=self.device,
            wrapper=False,
            rnn_mode="full_states",
            episode_dones=episode_dones,
            **self.recurrent_kwargs,
        )

        next_values = self.get_values(
            obs=next_obs, batch_size=batch_size, rnn_states=rnn_next_states, ret_device=self.device, wrapper=False, **self.recurrent_kwargs
        )

        if self.rew_rms is not None:
            std = self.rew_rms.std
            values = values * std
            next_values = next_values * std

        next_values = next_values * (1.0 - dones.float())

        delta = rewards + next_values * self.gamma - values

        coeff = episode_masks * self.gamma * self.lmbda
        advantages = torch.zeros(len(rewards), 1, device=self.device, dtype=torch.float32)

        gae = 0
        for i in range(len(rewards) - 1, -1, -1):
            gae = delta[i] + coeff[i] * gae
            advantages[i] = gae
        returns = advantages + values

        ret = {
            "old_values": values,
            "old_next_values": next_values,
            "original_returns": returns,
            "returns": returns,
            "advantages": advantages,
            # 'v_states': v_states,
            # 'v_next_states': v_next_states,
        }
        if self.rew_rms is not None:
            if update_rms:
                assert self.rew_rms.training
                self.rew_rms.add(ret["returns"])
            std = self.rew_rms.std

            ret["old_values"] = ret["old_values"] / std
            ret["old_next_values"] = ret["old_next_values"] / std
            ret["returns"] = ret["returns"] / std
        ret = GDict(ret).to_numpy()
        torch.cuda.empty_cache()
        return ret

    def actor_grad(self, with_shared=True):
        ret = {}
        if getattr(self, "actor", None) is None:
            return ret
        ret[f"grad/actor_grad_norm"] = self.actor.grad_norm
        if with_shared:
            assert self.shared_
            if getattr(self.actor_grad.backbone, "visual_nn", None) is not None:
                ret["grad/visual_grad"] = self.actor.backbone.visual_nn.grad_norm
            from pyrl.networks import RNN

            if getattr(self.actor.backbone, "rnn", None) is not None and not isinstance(self.actor.backbone, RNN):
                ret["grad/rnn_grad"] = self.actor.backbone.rnn.grad_norm
            if self.actor.final_mlp is not None:
                ret["grad/mlp_grad"] = self.actor.final_mlp.grad_norm

    def critic_grad(self, with_shared=True):
        ret = {}
        if getattr(self, "critic", None) is None:
            return ret
        ret[f"grad/critic_grad_norm"] = self.critic.grad_norm
        if with_shared:
            assert self.shared_
            if getattr(self.actor_grad.backbone, "visual_nn", None) is not None:
                ret["grad/visual_grad"] = self.actor.backbone.visual_nn.grad_norm
            from pyrl.networks import RNN

            if getattr(self.actor.backbone, "rnn", None) is not None and not isinstance(self.actor.backbone, RNN):
                ret["grad/rnn_grad"] = self.actor.backbone.rnn.grad_norm
            if self.actor.final_mlp is not None:
                ret["grad/mlp_grad"] = self.actor.final_mlp.grad_norm

    def to_normal(self):
        if self._be_data_parallel and self._device_ids is not None:
            self._be_data_parallel = False
            for module_name in dir(self):
                item = getattr(self, module_name)
                if isinstance(item, DDP):
                    setattr(self, module_name, item.module)


def async_no_grad_pi(pi):
    import torch

    def run(*args, **kwargs):
        with torch.no_grad():
            return pi(*args, **kwargs)

    return run
