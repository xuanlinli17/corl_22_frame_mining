import numpy as np
from pyrl.utils.data import DictArray, GDict
from pyrl.utils.meta import get_logger

from .builder import SAMPLING


@SAMPLING.register_module()
class SamplingStrategy:
    def __init__(self, with_replacement=True, capacity=None, no_random=False):
        self.with_replacement = with_replacement
        self.no_random = no_random
        if no_random:
            assert not with_replacement, "Fix order only supports without-replacement!"
        self.horizon = 1
        self.capacity = capacity  # same as replay buffer capacity, i.e. # of 1-steps
        self.position = 0
        self.running_count = 0

        # For without replacement
        self.items = None
        self.item_index = 0
        self.need_update = False

    def get_index(self, batch_size, capacity=None, drop_last=True, auto_restart=True):
        if capacity is None:  # For 1-Step Transition, capacity is the number of data samples
            capacity = len(self)
        # For T step transition, capacity is the number of valid trajectories, and should be specified in the capacity arg
        if self.with_replacement:
            return np.random.randint(low=0, high=capacity, size=batch_size)

        if self.items is None or self.need_update:
            self.need_update = False
            self.items = np.arange(capacity)
            if not self.no_random:
                np.random.shuffle(self.items)
            self.item_index = 0
        min_query_size = batch_size if drop_last else 1
        if self.item_index + min_query_size > capacity:
            if not auto_restart:
                return None
            if not self.no_random:
                np.random.shuffle(self.items)
            self.item_index = 0
        else:
            batch_size = min(batch_size, capacity - self.item_index)
        index = self.items[self.item_index : self.item_index + batch_size]
        self.item_index += batch_size
        return index

    def __len__(self):
        return min(self.running_count, self.capacity)

    def restart(self):
        self.item_index = 0
        self.items = None

    def reset_all(self):
        raise NotImplementedError

    def push_batch(self, items: DictArray):
        raise NotImplementedError

    def push(self, item: GDict):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError


@SAMPLING.register_module()
class OneStepTransition(SamplingStrategy):
    # Sample 1-step transitions. A special case of TStepTransition with better speed.
    def __init__(self, **kwargs):
        super(OneStepTransition, self).__init__(**kwargs)

    def reset(self):
        self.position = 0
        self.running_count = 0
        self.restart()

    def push_batch(self, items):
        self.need_update = True
        self.running_count += len(items)
        self.position = (self.position + len(items)) % self.capacity

    def push(self, item):
        self.need_update = True
        self.running_count += 1
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, drop_last=True, auto_restart=True):
        # Return: index and valid masks
        index = self.get_index(batch_size, len(self), drop_last=drop_last, auto_restart=auto_restart)
        if index is None:
            return None, None
        else:
            return index, np.ones([index.shape[0], self.horizon], dtype=np.bool)
