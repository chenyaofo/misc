import os
import torch
from .base import Plugin
from ..engine import Engine, Context, Event
from torchlearning.logging import experiment_path


# save model when satisfy specific condition
class Saver(Plugin):
    def __init__(self, condition, prefix="net", exclude=True, event=Event.STAGE_COMPLETED):
        super(Saver, self).__init__()
        self.condition = condition
        self.prefix = prefix
        self.exclude = exclude
        self.event = event
        self.last_save = None

    def save(self, ctx: Context):
        if self.condition(ctx):
            if ctx.is_last_iteration:
                pure_name = "{}@e{}.pth".format(self.prefix, ctx.epoch)
            else:
                pure_name = "{}@e{}i{}.pth".format(self.prefix, ctx.epoch, ctx.iteration)
            save_path = os.path.join(experiment_path, pure_name)
            torch.save(ctx.net.state_dict(), save_path)

            if self.exclude and self.last_save is not None:
                os.remove(self.last_save)
            self.last_save = save_path

    def attach(self, engine: Engine):
        engine.ctx.plugins["saver"] = self
        engine.add_event_handler(self.event, self.save)
