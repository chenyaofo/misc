from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from .stage import Stage
import torch


@dataclass(order=False)
class Context:
    epoch: int = field(default=0)
    max_epoch: int = field(default=None)
    iteration: int = field(default=0)
    max_iteration: int = field(default=0)
    net: torch.nn.Module = field(default=None)
    device: torch.device = field(default=None)
    train_loader: DataLoader = field(default=None)
    val_loader: DataLoader = field(default=None)
    test_loader: DataLoader = field(default=None)
    stage: Stage = field(default=None)
    optimizer: torch.optim.Optimizer = field(default=None)
    criterion: torch.nn.Module = field(default=None)
    scheduler: torch.optim.lr_scheduler._LRScheduler = field(default=None)
    meters: dict = field(default_factory=dict)
    plugins: dict = field(default_factory=dict)

    @property
    def is_training(self):
        return self.stage == Stage.TRAIN

    @property
    def is_validating(self):
        return self.stage == Stage.VALIDATION

    @property
    def is_testing(self):
        return self.stage == Stage.TEST

    @property
    def is_first_epoch(self):
        return self.epoch == 1

    @property
    def is_last_epoch(self):
        return self.epoch == self.max_epoch

    @property
    def is_first_iteration(self):
        return self.iteration == 1

    @property
    def is_last_iteration(self):
        return self.iteration == self.max_iteration
