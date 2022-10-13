"""
Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/checkpoint.py
"""

import os
import os.path as osp
import pkgutil
from collections import OrderedDict
from importlib import import_module

import numpy as np
import torchvision
from pyrl.utils.data import GDict, map_dict_keys

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils import model_zoo

from .misc import no_grad


@no_grad
def load_state_dict(module, state_dict, strict=False, logger=None):
    logger = logger.warning if logger is not None else print
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    for name, parameter in module.named_parameters():
        if name in state_dict and state_dict[name].shape != parameter.shape:
            # We only deal with different in_channel cases.
            soure_shape = np.array(state_dict[name].shape, dtype=np.int)
            target_shape = np.array(parameter.shape, dtype=np.int)
            if (soure_shape != target_shape).sum() == 1:
                logger(f"We adapt weight with shape {soure_shape} to shape {target_shape}.")
                tmp = parameter.data.clone()
                index = np.nonzero(soure_shape != target_shape)[0][0]
                tmp = tmp.transpose(0, index)
                num = min(soure_shape[index], target_shape[index])
                tmp[:num] = state_dict[name].transpose(0, index)[:num]
                state_dict[name] = tmp.transpose(0, index).detach().contiguous()

    loaded_modules = {}
    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        nonlocal loaded_modules
        if module in loaded_modules:
            return
        loaded_modules[module] = True
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        included_optimizer = []
        for name, child in module.__dict__.items():
            optimizer_name = f"{prefix}{name}"
            if child is not None and isinstance(child, Optimizer) and id(child) not in included_optimizer:
                if optimizer_name in state_dict:
                    included_optimizer.append(id(child))
                    """
                    print(optimizer_name, prefix, name)  # , state_dict[optimizer_name])

                    groups = child.param_groups
                    saved_groups = state_dict[optimizer_name]["param_groups"]
                    print(type(groups[0]["params"]))
                    from IPython import embed

                    embed()
                    param_lens = [len(g["params"]) for g in groups]
                    saved_lens = [len(g["params"]) for g in saved_groups]
                    print(param_lens, saved_lens)
                    exit(0)
                    """
                    try:
                        child.load_state_dict(state_dict.pop(optimizer_name))
                    except:
                        logger(f"We cannot load optimizer {optimizer_name}!")
                else:
                    included_optimizer.append(id(child))
                    logger(f"missing keys in source state_dict for optimizer {optimizer_name}")

        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            logger(err_msg)




def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).
    Args:
        filename (str): Accept local filepath, URL,
            ``torchvision://xxx``, ``open-mmlab://xxx``.
        Please refer to ``docs/model_zoo.md`` for details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.
    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if not osp.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model, filename, map_location=None, strict=False, keys_map=None, logger=None):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if keys_map is not None:
        state_dict = map_dict_keys(state_dict, keys_map, logger.info if logger is not None else None)

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
    # load state_dict
    if not isinstance(model, DDP):
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
    load_state_dict(model, state_dict, strict, logger)

    # exit(0)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = GDict(val).cpu(wrapper=False)
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix="", keep_vars=False):
    """Returns a dictionary containing a whole state of the module, including the state_dict of the optimizer in the moudle."""

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)

    included_optimizer = []
    for name, child in module.__dict__.items():
        if child is not None and isinstance(child, Optimizer) and id(child) not in included_optimizer:
            included_optimizer.append(id(child))
            destination[f"{prefix}{name}"] = child.state_dict()
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + ".", keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.
    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and ``optimizer``.
    By default ``meta`` will contain version and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f"meta must be a dict or None, but got {type(meta)}")

    os.makedirs(osp.dirname(filename), exist_ok=True)

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(get_state_dict(model))}
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint["optimizer"] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint["optimizer"] = {}
        for name, optim in optimizer.items():
            checkpoint["optimizer"][name] = optim.state_dict()

    # immediately flush buffer
    with open(filename, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
