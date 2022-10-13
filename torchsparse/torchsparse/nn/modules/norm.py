import torch
from torch import nn

from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase

__all__ = ['BatchNorm', 'GroupNorm', 'SyncBatchNorm', 'LayerNorm']


class BatchNorm(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: SparseTensor) -> SparseTensor:
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return fapply(
            input,
            F.batch_norm,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        if input.feats.dim() != 2 and input.feats.dim() != 3:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(input.feats.dim()))

    
class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """
        Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#SyncBatchNorm
        """
        module_output = module
        if isinstance(module, BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output


class GroupNorm(nn.GroupNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride

        batch_size = torch.max(coords[:, -1]).item() + 1
        num_channels = feats.shape[1]

        # PyTorch's GroupNorm function expects the input to be in (N, C, *)
        # format where N is batch size, and C is number of channels. "feats"
        # is not in that format. So, we extract the feats corresponding to
        # each sample, bring it to the format expected by PyTorch's GroupNorm
        # function, and invoke it.
        nfeats = torch.zeros_like(feats)
        for k in range(batch_size):
            indices = coords[:, -1] == k
            bfeats = feats[indices]
            bfeats = bfeats.transpose(0, 1).reshape(1, num_channels, -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(num_channels, -1).transpose(0, 1)
            nfeats[indices] = bfeats

        output = SparseTensor(coords=coords, feats=nfeats, stride=stride)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output


class LayerNorm(nn.LayerNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride
        out_feats = super(LayerNorm, self).forward(feats)
        output = SparseTensor(coords=coords, feats=out_feats, stride=stride)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

