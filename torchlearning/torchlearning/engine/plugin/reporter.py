import torchlearning as tl
import uuid
import torch
from .base import Plugin
from ..engine import Engine, Context, Event
from torchlearning.logging import experiment_path
import time


def aistr(obj):
    if obj is None:
        return "None"
    elif isinstance(obj, torch.optim.Optimizer):
        defaults = obj.defaults
        params_str = ["{}={}".format(k, v) for k, v in defaults.items()]
        return f"{obj.__class__.__name__}({', '.join(params_str)})"
    elif isinstance(obj, torch.optim.lr_scheduler._LRScheduler):
        d = obj.__dict__
        params_str = ["{}={}".format(k, v) for k, v in d.items() if k != "optimizer"]
        return f"{obj.__class__.__name__}({', '.join(params_str)})"
    elif isinstance(obj, torch.utils.data.DataLoader):
        d = obj.__dict__
        return "(size={},batch_size={},num_workers={},drop_last={})".format(
            len(obj.dataset),
            d["batch_size"],
            d["num_workers"],
            d["drop_last"]
        )
    else:
        return str(obj)


class Recorder(Plugin):
    def __init__(self, logger=tl.logger):
        super(Recorder, self).__init__()
        self.id= uuid.uuid4().hex
        self.logger = logger
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.net = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        self.meters = None
        self.plugins = None

    def close(self, *args, **kwargs):
        self.end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def attach(self, engine: Engine):
        engine.ctx.plugins["recorder"] = self
        engine.add_event_handler(Event.COMPLETED, self.close)
        self.net = aistr(engine.ctx.net)
        if hasattr(engine.ctx.train_loader.dataset,"name"):
            self.dataset = engine.ctx.train_loader.dataset.name
        else:
            self.dataset = "None"
        if engine.ctx.train_loader is not None:
            self.train_loader = "TrainLoader" + aistr(engine.ctx.train_loader)
        else:
            self.train_loader="None"
        if engine.ctx.val_loader is not None:
            self.val_loader = "ValidationLoader" + aistr(engine.ctx.val_loader)
        else:
            self.val_loader = "None"
        if engine.ctx.test_loader is not None:
            self.test_loader = "TestLoader" + aistr(engine.ctx.test_loader)
        else:
            self.test_loader = "None"
        self.device = aistr(engine.ctx.device)
        self.optimizer = aistr(engine.ctx.optimizer)
        self.criterion = aistr(engine.ctx.criterion)
        self.scheduler = aistr(engine.ctx.scheduler)

        self.meters = [v.__class__.__name__ for k, v in engine.ctx.meters.items()]
        self.plugins = [v.__class__.__name__ for k, v in engine.ctx.plugins.items()]
