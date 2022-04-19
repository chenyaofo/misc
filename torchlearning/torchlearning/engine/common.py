import torch

from .context import Context

def default_epoch_flow(engine):
    engine.train()
    engine.validate()

def default_train(ctx: Context):
    datas, targets = ctx.batch
    datas, targets = datas.to(ctx.device), targets.to(ctx.device)
    outputs = ctx.net(datas)
    loss = ctx.criterion(outputs, targets)
    ctx.optimizer.zero_grad()
    loss.backward()
    ctx.optimizer.step()
    ctx.targets = targets
    ctx.outputs = outputs
    ctx.loss = loss


def default_validate(ctx: Context):
    datas, targets = ctx.batch
    datas, targets = datas.to(ctx.device), targets.to(ctx.device)
    with torch.no_grad():
        outputs = ctx.net(datas)
        loss = ctx.criterion(outputs, targets)
    ctx.targets = targets
    ctx.outputs = outputs
    ctx.loss = loss
