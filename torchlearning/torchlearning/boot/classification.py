import torchlearning as tl
from torchlearning.engine import Engine, Context,Event
from torchlearning.engine.plugin import TimeEstimater,Saver
from torchlearning.meter import TimeMeter,AccuracyMeter,LossMeter
from torchlearning.logging.utils import report_machine_info


def last_save(ctx:Context):
    return ctx.is_last_epoch and ctx.is_validating

class ClassificationEngine(Engine):
    def __init__(self, ctx: Context):
        super(ClassificationEngine, self).__init__(ctx)

        report_machine_info()

        TimeMeter().attach(self)
        AccuracyMeter(topk=(1, 5)).attach(self)
        LossMeter().attach(self)
        TimeEstimater().attach(self)
        Saver(condition=last_save, exclude=False).attach(self)

        @self.on(Event.STAGE_COMPLETED)
        def sche(ctx: Context):
            if ctx.is_training and ctx.scheduler is not None:
                ctx.scheduler.step()

        @self.on(Event.ITER_COMPLETED)
        def iter_log(ctx: Context):
            tl.logger.info(
                "{}, Epoch={}, Iter={}/{}, Loss={:.4f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%, ETA={:.2f}s".format(
                    ctx.stage.value.upper(),
                    ctx.epoch,
                    ctx.iteration,
                    ctx.max_iteration,
                    ctx.meters["loss"].value,
                    ctx.meters["accuracy"][1].rate * 100,
                    ctx.meters["accuracy"][5].rate * 100,
                    ctx.plugins["eta"].value,
                ))

        @self.on(Event.STAGE_COMPLETED)
        def state_log(ctx: Context):
            tl.logger.info(
                "{} Complete, Epoch={}, Loss={:.4f}, Accuracy@1={:.2f}%, Accuracy@5={:.2f}%, Eplased Time={:.2f}s".format(
                    ctx.stage.value.upper(),
                    ctx.epoch,
                    ctx.meters["loss"].value,
                    ctx.meters["accuracy"][1].rate * 100,
                    ctx.meters["accuracy"][5].rate * 100,
                    ctx.plugins["timer"].value,
                ))
