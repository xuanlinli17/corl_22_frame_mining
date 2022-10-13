import numpy as np
from typing import Union
from itertools import count

from pyrl.utils.data import DictArray, GDict
from .builder import REPLAYS


@REPLAYS.register_module()
class ReplayMemory:

    def __init__(
        self,
        capacity,
        sampling_cfg=dict(type="OneStepTransition"),
    ):
        self.capacity = capacity
        self.memory = None
        self.position = 0
        self.running_count = 0
        self.reset()
        from .builder import build_sampling
        if sampling_cfg is not None:
            sampling_cfg["capacity"] = capacity
            self.sampling = build_sampling(sampling_cfg)

    def __getitem__(self, key):
        return self.memory[key]

    def __setitem__(self, key, value):
        self.memory[key] = value

    def __getattr__(self, key):
        return getattr(self.memory, key, None)

    def __len__(self):
        return min(self.running_count, self.capacity)

    def reset(self):
        self.position = 0
        self.running_count = 0
        # self.memory = None
        if self.sampling is not None:
            self.sampling.reset()

    def push(self, item):
        if not isinstance(item, DictArray):
            item = DictArray(item, capacity=1)
        self.push_batch(item)

    def push_batch(self, items: Union[DictArray, dict]):
        if not isinstance(items, DictArray):
            items = DictArray(items)
        if len(items) > self.capacity:
            items = items.take(slice(0, self.capacity))
            
        if "worker_index" not in items:
            items["worker_index"] = np.zeros([len(items), 1], dtype=np.int32)
        if "is_truncated" not in items:
            items["is_truncated"] = np.zeros([len(items), 1], dtype=np.bool)

        if self.memory is None:
            # Init the whole buffer
            self.memory = DictArray(items.take(0), capacity=self.capacity)

        if self.position + len(items) > self.capacity:
            # Deal with buffer overflow
            final_size = self.capacity - self.position
            self.push_batch(items.take(slice(0, final_size)))
            self.position = 0
            self.push_batch(items.take(slice(final_size, len(items))))
        else:
            self.memory.assign(slice(self.position, self.position + len(items)), items)
            self.running_count += len(items)
            self.position = (self.position + len(items)) % self.capacity
            if self.sampling is not None:
                self.sampling.push_batch(items)

    def update_all_items(self, items):
        self.memory.assign(slice(0, len(items)), items)

    def tail_mean(self, num):
        return self.memory.take(slice(len(self) - num, len(self))).to_gdict().mean()

    def get_all(self):
        # Return all elements in replay buffer
        return self.memory.take(slice(0, len(self)))

    def to_hdf5(self, file, with_traj_index=False):
        data = self.get_all()
        if with_traj_index:
            data = GDict({"traj_0": data.memory})
        data.to_hdf5(file)

    def sample(self, batch_size, auto_restart=True, drop_last=True):
        if self.dynamic_loading and not drop_last:
            assert self.capacity % batch_size == 0

        batch_idx, is_valid = self.sampling.sample(batch_size, drop_last=drop_last, auto_restart=auto_restart and not self.dynamic_loading)
        if batch_idx is None:
            # without replacement only
            if auto_restart or self.dynamic_loading:
                items = self.file_loader.get()
                if items is None:
                    return None
                assert self.position == 0, "cache size should equals to buffer size"

                # st = time.time()
                self.sampling.reset()
                self.push_batch(items)
                self.file_loader.run(auto_restart=auto_restart)
                # print('Get next items', time.time() - st)
                batch_idx, is_valid = self.sampling.sample(batch_size, drop_last=drop_last, auto_restart=auto_restart and not self.dynamic_loading)
            else:
                return None

        ret = self.memory.take(batch_idx)
        ret["is_valid"] = is_valid
        return ret

    def mini_batch_sampler(self, batch_size, drop_last=False, auto_restart=False, max_num_batches=-1):
        if self.sampling is not None:
            old_replacement = self.sampling.with_replacement
            self.sampling.with_replacement = False
            self.sampling.restart()
        for i in count(1):
            if i > max_num_batches and max_num_batches != -1:
                break
            items = self.sample(batch_size, auto_restart, drop_last)
            if items is None:
                self.sampling.with_replacement = old_replacement
                break
            yield items
