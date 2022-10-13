from functools import wraps
import numpy as np
import torch
from pyrl.utils.math import split_num
from pyrl.utils.meta import get_logger
from pyrl.utils.data import DictArray, to_np, GDict, to_torch


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


def auto_torch(items):

    pass


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed. Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


def no_grad(f):
    wraps(f)

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return wrapper


def get_seq_info(done_mask):
    # It will sort the length of the sequence to improve the performance

    # input: done_mask [L]
    # return: index [#seq, max_seq_len]; sorted_idx [#seq]; is_valid [#seq, max_seq_len]
    done_mask = to_np(done_mask)

    indices = np.where(done_mask)[0]
    one = np.ones(1, dtype=indices.dtype)
    indices = np.concatenate([one * -1, indices])
    len_seq = indices[1:] - indices[:-1]

    sorted_idx = np.argsort(-len_seq, kind="stable")  # From max to min
    max_len = len_seq[sorted_idx[0]]
    index = np.zeros([len(sorted_idx), max_len], dtype=np.int64)
    is_valid = np.zeros([len(sorted_idx), max_len], dtype=np.bool)

    for i, idx in enumerate(sorted_idx):
        index[i, : len_seq[idx]] = np.arange(len_seq[idx]) + indices[idx] + 1
        is_valid[i, : len_seq[idx]] = True
    return index, sorted_idx, is_valid


def run_with_mini_batch(
    function,
    *args,
    batch_size=None,
    wrapper=True,
    device=None,
    ret_device=None,
    episode_dones=None,
    is_recurrent=False,
    recurrent_horizon=-1,
    rnn_mode="base",
    **kwargs,
):
    """
    Run a pytorch function with mini-batch when the batch size of dat is very large.
    :param function: the function
    :param data: the input data which should be in dict array structure
    :param batch_size: the num of samples in the whole batch, it is num of episodes * episode length for recurrent data.
    :return: all the outputs.
    """
    capacity = None
    assert rnn_mode in ["base", "full_states", "with_states"], f"{rnn_mode} is not supported by rnn_mode"
    # print('In mini batch', batch_size, DictArray(kwargs).shape)

    def process_kwargs(x):
        if x is None or len(x) == 0:
            return None
        # print(type(x))
        nonlocal capacity, device, ret_device
        x = DictArray(x)
        # print(x.shape, x.type)
        # exit(0)
        capacity = x.capacity
        if device is None:
            device = x.one_device
        if ret_device is None:
            ret_device = x.one_device
        return x

    args, kwargs = list(args), dict(kwargs)
    # print(GDict(args).type)
    # print(GDict(kwargs).type)

    args = process_kwargs(args)
    kwargs = process_kwargs(kwargs)

    assert capacity is not None, "Input is None"
    if batch_size is None:
        batch_size = capacity
    recurrent_run = (episode_dones is not None) and is_recurrent
    # print(type(episode_dones), recurrent_run, is_recurrent)
    # exit(0)
    if not recurrent_run:
        ret = []
        # print(capacity, batch_size)
        for i in range(0, capacity, batch_size):
            num_i = min(capacity - i, batch_size)
            args_i = args.take(slice(i, i + num_i)).to_torch(device=device, wrapper=False) if args is not None else []
            kwargs_i = kwargs.take(slice(i, i + num_i)).to_torch(device=device, wrapper=False) if kwargs is not None else {}
            # print('Iter', i, function, GDict(args_i).device, GDict(kwargs_i).device, device, ret_device)
            # exit(0)
            if rnn_mode != "base":
                kwargs_i["rnn_mode"] = rnn_mode
            ret.append(GDict(function(*args_i, **kwargs_i)).to_torch(device=ret_device, wrapper=False))
        ret = DictArray.concat(ret, axis=0, wrapper=wrapper)
        if rnn_mode != "base":
            # print(GDict(ret).type)
            # exit(0)
            ret, states = ret
    else:
        assert episode_dones is not None and rnn_mode in ["base", "full_states"], f"Flags {episode_dones is not None}, {rnn_mode}"

        if kwargs is not None:
            assert kwargs.memory.pop("rnn_states", None) is None, "You do not need to provide rnn_states!"
        index, sorted_index, is_valid = get_seq_info(episode_dones)

        capacity = len(index)
        ret = [None for i in range(capacity)]
        current_states = [None for i in range(capacity)]
        next_states = [None for i in range(capacity)]
        batch_size = batch_size // recurrent_horizon

        # get_logger().info(f"{capacity, batch_size, recurrent_horizon, index.shape}")

        for i in range(0, capacity, batch_size):
            # Batch over trajectories and then Batch over horizon
            num_i = min(capacity - i, batch_size)
            index_i = index[i : i + num_i]
            is_valid_i = to_torch(is_valid[i : i + num_i], device=device)
            max_len = is_valid_i[0].sum().item()

            is_valid_i = is_valid_i[:, :max_len]
            args_i = args.take(index_i).to_torch(device=device) if args is not None else None
            kwargs_i = kwargs.take(index_i).to_torch(device=device) if kwargs is not None else None

            # kwargs_i['is_valid'] = is_valid_i
            rnn_states = None
            tmp = []

            # get_logger().info(f"{index_i.shape, i, max_len, GDict([args_i, kwargs_i]).shape}")
            # print(max_len, recurrent_horizon, batch_size, capacity)

            for j in range(0, max_len, recurrent_horizon):
                num_j = min(max_len - j, recurrent_horizon)
                args_ij = args_i.take(slice(j, j + num_j), axis=1, wrapper=False) if args_i is not None else []
                kwargs_ij = kwargs_i.take(slice(j, j + num_j), axis=1, wrapper=False) if kwargs_i is not None else {}
                is_valid_ij = is_valid_i[:, j : j + num_j]

                # print(GDict(rnn_states).shape, type(rnn_states))

                ret_ij, rnn_states_all = function(*args_ij, **kwargs_ij, rnn_states=rnn_states, is_valid=is_valid_ij, rnn_mode="full_states")
                rnn_states_all, rnn_states = rnn_states_all[:-1], rnn_states_all[-1]

                ret_ij, rnn_states_all = GDict([ret_ij, rnn_states_all]).to_torch(device=ret_device, wrapper=False)
                tmp.append([ret_ij, rnn_states_all])
            tmp = DictArray.concat(tmp, axis=1, wrapper=False)
            # print('Tmp', GDict(tmp).shape)
            ret_i, [current_states_i, next_states_i] = tmp
            ret_i = DictArray(ret_i)
            current_states_i = DictArray(current_states_i)
            next_states_i = DictArray(next_states_i)

            for j, ori_idx in enumerate(sorted_index[i : i + num_i]):
                # print(ret_i.memory[0])
                # exit(0)
                ret[ori_idx] = ret_i.take(j).select_with_mask(is_valid_i[j], wrapper=False)
                # print(j, ori_idx, ret[ori_idx].shape)
                # print(current_states_i.shape, is_valid_i.shape)
                current_states[ori_idx] = current_states_i.take(j).select_with_mask(is_valid_i[j], wrapper=False)
                # print(j, ori_idx, current_states[ori_idx].shape)

                # print(next_states_i.shape, is_valid_i.shape)
                next_states[ori_idx] = next_states_i.take(j).select_with_mask(is_valid_i[j], wrapper=False)
                # print(j, ori_idx, next_states[ori_idx].shape)

                # get_logger().info(f"{j, ori_idx, is_valid_i.shape, ret[ori_idx].shape}")
        ret = DictArray.concat(ret, axis=0, wrapper=wrapper)
        current_states = DictArray.concat(current_states, axis=0, wrapper=wrapper)
        next_states = DictArray.concat(next_states, axis=0, wrapper=wrapper)
        states = [current_states, next_states, None]
    return [ret, states] if rnn_mode != "base" else ret


def mini_batch(wrapper_=True):
    def actual_mini_batch(f):
        wraps(f)

        def wrapper(*args, batch_size=None, wrapper=None, device=None, ret_device=None, **kwargs):
            if wrapper is None:
                wrapper = wrapper_
            # print(batch_size, dict(kwargs))

            return run_with_mini_batch(f, *args, **kwargs, batch_size=batch_size, wrapper=wrapper, device=device, ret_device=ret_device)

        return wrapper

    return actual_mini_batch
