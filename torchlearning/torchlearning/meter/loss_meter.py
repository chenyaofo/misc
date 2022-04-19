import torch
from .meter import Meter
from torchlearning.engine import Engine, Event


class LossMeter(Meter):
    def __init__(self):
        super(LossMeter, self).__init__()
        self.reset()

    def reset(self, *args, **kwargs):
        self.n = 0
        self.loss = 0.

    def update(self, loss):
        if torch.is_tensor(loss):
            self.n += 1
            self.loss += loss.item()
        else:
            raise ValueError("'loss' should be torch.tensor(scalar), but found {}"
                             .format(type(loss)))

    def _engine_update(self, ctx):
        loss = ctx.loss
        self.update(loss)

    def attach(self, engine: Engine):
        engine.ctx.meters["loss"] = self
        engine.add_event_handler(Event.STAGE_STARTED, self.reset)
        engine.add_event_handler(Event.ITER_COMPLETED, self._engine_update)

    @property
    def value(self):
        if self.n == 0:
            return 0.0
        return self.loss / self.n

    @property
    def record(self):
        return dict(loss=self.value)

    def __str__(self):
        return "Loss={:.4f}".format(self.value())
