import math
import torchlearning as tl

from torchlearning.engine import Engine, Event, Stage, Context

from scipy.ndimage import uniform_filter

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import numpy

plt.switch_backend('agg')


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class LearningRateSearchEngine(object):
    def __init__(self, loader, net, optimizer, criterion, device):
        self.ctx = Context()
        self.ctx.train_loader = loader
        self.ctx.net = net
        self.ctx.optimizer = optimizer
        self.ctx.criterion = criterion
        self.ctx.device = device
        self.ctx.lrs = []
        self.ctx.losses = []

        self.engine = Engine(self.ctx)

        @self.engine.core_function(Stage.TRAIN)
        def cycle_learning_rate_train(ctx: Context):
            # calculate for current learning rate
            power_lower_limit = math.log(ctx.lr_min, 10)
            power_upper_limit = math.log(ctx.lr_max, 10)
            current_power = power_lower_limit + (
                    power_upper_limit - power_lower_limit) / ctx.max_iteration * ctx.iteration
            lr = math.pow(10, current_power)
            ctx.lr = lr
            ctx.lrs.append(lr)
            set_learning_rate(ctx.optimizer, lr=lr)

            # forward and backward
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

            # record loss scaler
            ctx.losses.append(loss.item())

        @self.engine.on(Event.ITER_COMPLETED)
        def log2console(ctx: Context):
            tl.logger.info("Cycle Learning Rate Search, Iter={}/{}, Learning Rate={:.8f}, Loss={:.8f}.".format(
                ctx.iteration,
                ctx.max_iteration,
                ctx.lr,
                ctx.losses[-1],
            ))

    def set_lr_search_scope(self, min, max):
        self.ctx.lr_min = min
        self.ctx.lr_max = max

    def run(self):
        self.engine._run_once(Stage.TRAIN)

    def to_figure(self, save_path, uniform_filter_size=11, loss_clip_max=10):
        pp = PdfPages(save_path)
        fig = plt.figure(figsize=(16, 9))
        losses = numpy.array(self.ctx.losses).clip(max=loss_clip_max)
        lrs = numpy.array(self.ctx.lrs)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(lrs, losses, color="#969696", label="Original Loss")
        uniform_losses = uniform_filter(losses, uniform_filter_size)
        plt.plot(lrs, uniform_losses, color="#0066FF", label="Gaussian Smoothing Loss")
        plt.title("Loss varying with learning rate from {:.6f} to {:.6f}.".format(self.ctx.lr_min, self.ctx.lr_max))
        plt.ylabel("loss")
        plt.xlabel("learning rate")
        plt.legend(loc=2)
        ax.set_xscale('log')
        pp.savefig(fig)
        pp.close()
        plt.close()

    def to_online(self, uniform_filter_size=11, loss_clip_max=10):
        import os
        from torchlearning.logging import experiment_path
        pdf_path = os.path.join(experiment_path, "lrs.pdf")
        self.to_figure(pdf_path, uniform_filter_size, loss_clip_max)
        from flask import Flask, send_from_directory

        app = Flask("lrs_preview")

        @app.route("/lrs.pdf", methods=["GET"])
        def lrs_preview():
            return send_from_directory(experiment_path, "lrs.pdf", mimetype="application/pdf")

        from torchlearning.utils import get_host_ip
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((get_host_ip(), 0))
        port = sock.getsockname()[1]
        sock.close()

        print("Please click http://{}:{}/lrs.pdf to view the learning rate search result.".format(
            get_host_ip(), port,
        ))

        app.run(host=get_host_ip(), port=port)
