import torch

from torchlearning.meter import LossMeter

class TestLossMeter(object):
    def test_init(self):
        loss_meter = LossMeter()
        assert loss_meter.value == 0.0

    def test_value(self):
        loss_meter = LossMeter()
        loss_meter.update(torch.tensor(1.0))
        assert loss_meter.value == 1.0

    def test_add(self):
        loss_meter = LossMeter()
        loss_meter.update(torch.tensor(1.0))
        loss_meter.update(torch.tensor(2.0))
        assert loss_meter.value == 1.5
        assert loss_meter.record["loss"] == 1.5

