import torch
from .meter import Meter
from torchlearning.engine import Engine, Event


class ConfusionMatrixMeter(Meter):
    def __init__(self, n_classes):
        super(ConfusionMatrixMeter, self).__init__()
        self.n_classes = n_classes
        self.confusion_matrix = None
        self.reset()

    def reset(self, *args, **kwargs):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int32)

    def update(self, targets, outputs):
        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.view(-1)

        self.confusion_matrix += torch.bincount(pred + self.n_classes * targets, minlength=self.n_classes ** 2) \
            .to(dtype=torch.int32) \
            .view(self.n_classes, self.n_classes).cpu()

    def _engine_update(self, ctx):
        if ctx.is_validating:
            targets = ctx.targets
            outputs = ctx.outputs
            self.update(targets, outputs)

    def attach(self, engine):
        engine.ctx.meters["confusion_matrix"] = self
        engine.add_event_handler(Event.STAGE_STARTED, self.reset)
        engine.add_event_handler(Event.ITER_COMPLETED, self._engine_update)

    @property
    def value(self):
        return self.confusion_matrix

    @property
    def record(self):
        return None
