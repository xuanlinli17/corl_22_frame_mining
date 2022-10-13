"""
LayerNormLSTM is copied from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
"""
import torch.nn as nn, torch, torch.jit as jit
from torch import Tensor
from torch.nn import Parameter
from typing import List, Tuple
from pyrl.utils.meta import Registry, build_from_cfg


RECURRENT_LAYERS = Registry("recurrent layer")


for module in [nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell]:
    RECURRENT_LAYERS.register_module(module=module)


def build_recurrent_layer(cfg):
    if cfg.get("type", None) == "TransformerXL":
        from ..builder import build_backbone

        return build_backbone(cfg)
    else:
        return build_from_cfg(cfg, RECURRENT_LAYERS)
