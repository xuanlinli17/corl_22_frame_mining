from .type_utils import is_h5


def deepcopy(item):
    from copy import deepcopy

    if not is_h5(item):
        item = deepcopy(item)
    return item


"""
def copy(item):
    from copy import copy
    if not is_h5(item):
        item = copy(item)
    return item
"""


def equal(x, y):
    return True if x is None or y is None else x == y
