import numpy as np


def get_mean_lr(optimizer, mean=True):
    ret = []
    for param_group in optimizer.param_groups:
        ret.append(param_group["lr"])
    return np.mean(ret)
