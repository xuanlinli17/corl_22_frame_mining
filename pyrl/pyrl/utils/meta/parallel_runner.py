from inspect import isfunction
from ctypes import c_bool, c_int32
from multiprocessing import Process, set_start_method, Pipe, Value, shared_memory
from copy import deepcopy
import numpy as np, random, time


try:
    set_start_method("spawn")
except RuntimeError:
    pass


class Worker(Process):
    ASK = 1
    CALL = 2
    GETATTR = 3
    CONTINUE = 4
    EXIT = 5

    def __init__(self, cls, worker_id, base_seed=None, daemon=True, mem_infos=None, is_class=True, *args, **kwargs):
        super(Process, self).__init__()
        self.worker_id = worker_id
        if worker_id is None or base_seed is None:
            self.seed = None
        else:
            self.seed = base_seed + worker_id

        self.cls = cls
        self.is_class = is_class
        self.args = deepcopy(args)
        self.kwargs = deepcopy(dict(kwargs))
        self.kwargs["worker_id"] = worker_id

        self.pipe, self.worker_pipe = Pipe(duplex=True)
        self.daemon = daemon

        self.initialized = Value(c_bool, False)
        self.running = Value(c_bool, False)
        self.item_in_pipe = Value(c_int32, 0)
        self.shared_memory = Value(c_bool, mem_infos is not None)
        self.mem_infos = mem_infos

        self.shared_mem = None
        if mem_infos is not None:
            self.input_mem = shared_memory.SharedMemory(create=True, size=1024**2)  # 1M input information
        self.len_input = Value(c_int32, 0)

        if hasattr(self, "start"):
            self.start()
        else:
            print("We should merge this class to another class")
            exit(0)

    def _return_results(self, ret):
        if self.shared_memory.value:
            self.shared_mem.assign(self.worker_id, ret)
            with self.item_in_pipe.get_lock():
                self.item_in_pipe.value += 1
            # Shared memory needs the object asignment to be finished
        else:
            # Send object will wait the object to be received!
            with self.item_in_pipe.get_lock():
                self.item_in_pipe.value += 1
            self.worker_pipe.send(ret)

    def run(self):
        from pyrl.utils.data import SharedDictArray
        from pyrl.utils.file import load

        if self.shared_memory.value:
            self.shared_mem = SharedDictArray(None, *self.mem_infos)

        if self.is_class:
            func = self.cls(*self.args, **self.kwargs)
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            if self.is_class and hasattr(func, "seed"):
                # For gym environment
                func.seed(self.seed)

        self.running.value = False
        with self.item_in_pipe.get_lock():
            self.item_in_pipe.value = 0

        while True:
            self.initialized.value = True
            if self.shared_memory.value:
                if self.len_input.value == 0:
                    continue
                op, args, kwargs = load(bytes(self.input_mem.buf[: self.len_input.value]), file_format="pkl")
                with self.len_input.get_lock():
                    self.len_input.value = 0
            else:
                op, args, kwargs = self.worker_pipe.recv()
            if op == self.CONTINUE:
                continue
            if op == self.EXIT:
                if func is not None and self.is_class:
                    del func
                self.worker_pipe.close()
                return
            self.running.value = True

            # print('1')
            if op == self.ASK:
                ret = func(*args, **kwargs)
            elif op == self.CALL:
                assert self.is_class
                func_name = args[0]
                args = args[1]
                ret = getattr(func, func_name)(*args, **kwargs)
            elif op == self.GETATTR:
                assert self.is_class
                ret = getattr(func, args)

            # print(op, func_name, args, kwargs, self.shared_memory.value, self.item_in_pipe.value, self.running.value)
            # print(ret)

            self.running.value = False
            self._return_results(ret)
            # print('GG')
            # print(self.item_in_pipe.value, self.running.value)
            # print(op, func_name, args, kwargs, self.shared_memory.value, self.item_in_pipe.value, self.running.value)

    def _send_info(self, info):
        """
        Executing some functions, before this we need to clean up the reaming results in pipe.
        It is important when we use async_get.
        """
        # print(self.shared_memory.value)
        # print(self.item_in_pipe.value)
        assert self.item_in_pipe.value in [
            0,
        ]
        if self.shared_memory.value:
            from pyrl.utils.file import dump

            info = dump(info, file_format="pkl")
            self.input_mem.buf[: len(info)] = info
            self.len_input.value = len(info)
        else:
            # if self.item_in_pipe.value > 0:
            #     assert self.item_in_pipe.value == 1
            #     self.pipe.recv()
            # print(info)
            self.pipe.send(info)

    def call(self, func_name, *args, **kwargs):
        self._send_info([self.CALL, [func_name, args], kwargs])

    def get_attr(self, attr_name):
        self._send_info([self.GETATTR, attr_name, None])

    def ask(self, *args, **kwargs):
        self._send_info([self.ASK, args, kwargs])

    @property
    def is_running(self):
        return self.running.value

    @property
    def is_idle(self):
        return not self.running.value and self.item_in_pipe.value == 0

    @property
    def is_ready(self):
        return not self.running.value and self.item_in_pipe.value > 0

    def set_shared_memory(self, value=True):
        if self.shared_memory.value == value:
            return
        self.shared_memory.value = value
        if value:
            self.pipe.send([self.CONTINUE, None, None])

    def wait(self, timeout=-1):
        st = None
        # cnt = 0
        while self.item_in_pipe.value < 1 or self.is_running:
            # cnt += 1
            # print(self.worker_id, self.item_in_pipe.value, self.running.value, self.initialized.value, cnt)
            if self.initialized.value and st is None:
                st = time.time()
            if st is not None and time.time() - st > timeout and timeout > 0:
                print(self.item_in_pipe.value < 1, self.running.value, self.initialized.value)
                raise RuntimeError(f"Nothing to get from pipe after {time.time() - st}s")

    def get_async(self):
        if self.item_in_pipe.value > 0:
            assert self.item_in_pipe.value == 1, f"{self.item_in_pipe.value}"
            ret = True if self.shared_memory.value else self.pipe.recv()
            with self.item_in_pipe.get_lock():
                self.item_in_pipe.value -= 1
        else:
            ret = None
        return ret

    def debug_print(self):
        print("Out", self.shared_memory.value, self.item_in_pipe.value, self.running.value)

    def get(self, timeout=-1):
        # if not self.shared_memory.value and timeout < 0:
        # timeout = 60.0
        # self.debug_print()

        # self.debug_print()
        # ret =  self.pipe.recv()
        # self.debug_print()

        # with self.item_in_pipe.get_lock():
        #     self.item_in_pipe.value -= 1
        # return ret
        self.wait(timeout)
        if not self.shared_memory.value:
            ret = self.pipe.recv()
            with self.item_in_pipe.get_lock():
                self.item_in_pipe.value -= 1
            return ret
        else:
            return self.get_async()

    def close(self):
        # self.pipe.close()
        # self.worker_pipe.close()
        try:
            self.input_mem.unlink()
            self.input_mem.close()
        except:
            pass

    def __del__(self):
        self.close()
        if self.is_alive():
            self.terminate()
        del self.pipe
        del self.worker_pipe
