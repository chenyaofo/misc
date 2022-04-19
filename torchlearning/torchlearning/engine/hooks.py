from torch.autograd import Variable
from torchlearning.meter import Meter
from torchlearning.meter.loss_meter import LossMeter
from torchlearning.meter.accuracy_meter import AccuracyMeter


def cudalize(state):
    inputs, targets = state["sample"]
    if state["engine"].cudable:
        inputs, targets = inputs.cuda(), targets.cuda()
    state.update(dict(sample=(inputs, targets)))


def variablize(state):
    inputs, targets = state["sample"]
    inputs, targets = Variable(inputs), Variable(targets)
    state.update(dict(sample=(inputs, targets)))


def network_specialize(state):
    if state["engine"]:
        state["network"].train()
    else:
        state["network"].eval()


def loss_meter_initialize(state):
    state.update(dict(loss_meter=LossMeter()))


def acc_meter_initialize(state):
    state.update(dict(acc_meter=AccuracyMeter()))

def loss_meter_update(state):
    state["loss_meter"].add_obeject(state["loss"])


def acc_meter_update(state):
    _,targets = state["sample"]
    state["acc_meter"].add_obeject(targets, state["output"])


def report_progress(state):
    strs = []
    strs.append("Epoch={}".format(state.get("epoch","VAL")))
    for name, item in state.items():
        if isinstance(item, Meter):
            strs.append(item.__str__())
    return print(" ".join(strs))