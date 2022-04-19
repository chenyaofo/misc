import torch.nn as nn
import torch.optim as optim
import itertools


def FineTuneSGD(net, lr, new_layers, new_layers_lr, momentum=0, dampening=0,
                weight_decay=0, nesterov=False):
    if isinstance(new_layers, nn.Module):
        new_layers = [new_layers]

    new_param_ids = [map(id, layer.parameters()) for layer in new_layers]
    new_params = filter(lambda p: id(p) in itertools.chain(new_param_ids), net.parameters())
    old_params = filter(lambda p: id(p) not in itertools.chain(new_param_ids), net.parameters())

    return optim.SGD(
        params=[
            dict(params=old_params, lr=lr),
            dict(params=new_params),
        ],
        lr=new_layers_lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov
    )