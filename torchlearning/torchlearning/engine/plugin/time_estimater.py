import time
from .base import Plugin
from ..engine import Engine, Event, Context


class TimeEstimater(Plugin):
    def __init__(self):
        super(TimeEstimater, self).__init__()

    def reset(self, *args, **kwargs):
        self.start = time.time()

    def update(self, current_progress):
        self.current_progress = current_progress

    def set_total(self, total_progress):
        self.total_progress = total_progress

    def _engine_update(self, ctx: Context):
        self.update(ctx.iteration)

    def _engine_set_total(self, ctx: Context):
        self.set_total(ctx.max_iteration)

    @property
    def value(self):
        now = time.time()
        remain = (now - self.start) / self.current_progress * (self.total_progress - self.current_progress)
        return remain

    def attach(self, engine: Engine):
        engine.ctx.plugins["eta"] = self
        engine.add_event_handler(Event.STAGE_STARTED, self.reset)
        engine.add_event_handler(Event.STAGE_STARTED, self._engine_set_total)
        engine.add_event_handler(Event.ITER_COMPLETED, self._engine_update)
