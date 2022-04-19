import torch

import logging

from .gpu_utils import GPUsMonitor

logger = logging.getLogger("torchlearning")


class GPUManagerHelper(object):
    def __init__(self):
        self.gm = GPUsMonitor()
        self.exclude_gpus = []

    def exclude(self, gpu):
        if isinstance(gpu, int):
            self.exclude_gpus = [gpu]
        elif isinstance(gpu, list):
            self.exclude_gpus = gpu
        else:
            raise ValueError("Exclude gpu should be a int or list of int, but find {}"
                             .format(type(gpu)))
        return self

    def auto_device(self):
        self.gm.update()
        if self.gm.nvml_init:
            to_select_gpu_ids = filter(lambda id_: not id_ in self.exclude_gpus, range(self.gm.n_gpus))
            max_free_memory = 0
            to_select_id = None
            for id_ in to_select_gpu_ids:
                if self.gm.gpus[id_].memory.free > max_free_memory:
                    max_free_memory = self.gm.gpus[id_].memory.free
                    to_select_id = id_
            if to_select_id is None:
                return torch.device("cpu")
            return torch.device("cuda:{}".format(to_select_id))
        else:
            return torch.device("cpu")


GPUManager = GPUManagerHelper()
