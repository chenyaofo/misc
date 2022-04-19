import torch

from collections import defaultdict

from .context import Context
from .event import Event
from .stage import Stage

from .common import default_train, default_validate, default_epoch_flow


class Engine(object):
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self._core_functions = dict()

        self._event_handlers = defaultdict(list)

        self._epoch_flow_function = default_epoch_flow

        self._loaders = dict()
        self._loaders[Stage.TRAIN] = self.ctx.train_loader
        self._loaders[Stage.VALIDATION] = self.ctx.val_loader
        self._loaders[Stage.TEST] = self.ctx.test_loader

        # set default core function
        self._core_functions[Stage.TRAIN] = default_train
        self._core_functions[Stage.VALIDATION] = default_validate

        # set epoch flow

    def set_max_epoch(self, max_epoch):
        self.ctx.max_epoch = max_epoch

    def core_function(self, stage):

        def decorator(f):
            self._core_functions[stage] = f
            return f

        return decorator

    def on(self, event):

        def decorator(f):
            self.add_event_handler(event, f)
            return f

        return decorator

    @property
    def epoch_flow(self):
        def decorator(f):
            self._epoch_flow_function = f
            return f

        return decorator

    def run(self):
        self._trigger_event(Event.STARTED)
        while self.ctx.epoch < self.ctx.max_epoch:
            self.ctx.epoch += 1
            self._trigger_event(Event.EPOCH_STARTED)
            self._epoch_flow_function(self)
            self._trigger_event(Event.EPOCH_COMPLETED)
        self._trigger_event(Event.COMPLETED)

    def train(self):
        self.ctx.stage = Stage.TRAIN
        self.ctx.net.train()
        self._run_once(Stage.TRAIN)

    def validate(self):
        self.ctx.stage = Stage.VALIDATION
        self.ctx.net.eval()
        self._run_once(Stage.VALIDATION)

    def test(self):
        self.ctx.stage = Stage.TEST
        self.ctx.net.eval()
        self._run_once(Stage.TEST)

    def add_event_handler(self, event: Event, handler):
        self._event_handlers[event].append(handler)

    def _run_once(self, stage: Stage, ):
        core_function = self._core_functions[stage]
        loader = self._loaders[stage]
        self.ctx.iteration = 0
        self.ctx.max_iteration = len(loader)

        self._trigger_event(Event.STAGE_STARTED)
        for batch in loader:
            self.ctx.iteration += 1
            self.ctx.batch = batch
            self._trigger_event(Event.ITER_STARTED)
            self.ctx.batch_result = core_function(self.ctx)
            self._trigger_event(Event.ITER_COMPLETED)
            self.ctx.batch = None
        self._trigger_event(Event.STAGE_COMPLETED)

    def _trigger_event(self, event: Event):
        handlers = self._event_handlers.get(event)
        if handlers is not None:
            for handler in handlers:
                handler(self.ctx)
