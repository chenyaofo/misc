import math

import numpy
from scipy.ndimage import uniform_filter

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages

from ..logging.trainrecorder import TrainRecorder

plt.switch_backend('agg')


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LRAnalyzer(object):
    def __init__(self, loader, net, optimizer, criterion):
        self.cudable = torch.cuda.is_available()
        self.optimizer = optimizer
        self.net = net
        self.loader = loader
        self.criterion = criterion
        self.n_iters = len(self.loader)
        self.history_loss = 0.
        self.recorder = TrainRecorder(name="LRAnalyze")

    def set_explore_scope(self, scope=(1e-5, 1)):
        if not isinstance(scope, tuple):
            raise Exception("scope should be a tuple consist of two float number.")
        self.scope = scope
        self.power_lower_limit = math.log(scope[0], 10)
        self.power_upper_limit = math.log(scope[1], 10)

    def run(self):
        for batch_index, (inputs, targets) in enumerate(self.loader, start=1):
            current_power = self.power_lower_limit + (self.power_upper_limit - self.power_lower_limit) / self.n_iters * batch_index
            print(current_power)
            set_learning_rate(self.optimizer,
                              lr=math.pow(10, current_power))

            self.recorder.add_scalar_summary(lr=self.optimizer.param_groups[0]["lr"])
            if self.cudable:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs_before = self.net(inputs)
            loss_before = self.criterion(outputs_before, targets)
            self.recorder.add_scalar_summary(loss_before=loss_before.data[0])
            self.optimizer.zero_grad()
            loss_before.backward()
            self.optimizer.step()

            outputs_after = self.net(inputs)
            loss_after = self.criterion(outputs_after, targets)
            self.recorder.add_scalar_summary(loss_after=loss_after.data[0])

    @staticmethod
    def analyze(recorder, save_path):
        pp = PdfPages(save_path)
        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.93, top=0.95, wspace=0.20, hspace=0.25)
        ax11 = plt.subplot(221)
        loss_before = numpy.array(recorder.loss_before)
        plt.plot(recorder.lr, loss_before, color="#969696")
        loss_before_ = uniform_filter(loss_before,23)
        plt.plot(recorder.lr, loss_before_, color="#0066FF")
        plt.title("Loss before optimization.")
        plt.xlabel("epoch")
        ax11.set_xscale('log')
        ax12 = plt.subplot(222)
        loss_after = recorder.loss_after
        plt.plot(recorder.lr, loss_after, color="#969696")
        loss_after_ = uniform_filter(loss_after, 23)
        plt.plot(recorder.lr, loss_after_, color="#0066FF")
        plt.title("Loss after optimization.")
        plt.xlabel("epoch")
        ax12.set_xscale('log')
        ax13 = plt.subplot(223)
        loss_diff = loss_after - loss_before
        plt.plot(recorder.lr, loss_diff, color="#969696")
        loss_diff_ = uniform_filter(loss_diff, 23)
        plt.plot(recorder.lr, loss_diff_, color="#0066FF")
        plt.title("Loss diff.")
        plt.xlabel("epoch")
        ax13.set_xscale('log')
        ax14 = plt.subplot(224)
        loss_per_diff = 100*loss_diff / loss_before
        plt.plot(recorder.lr, loss_per_diff, color="#969696")
        loss_per_diff = uniform_filter(loss_per_diff, 23)
        plt.plot(recorder.lr, loss_per_diff, color="#0066FF")
        plt.title("Loss persent diff.")
        plt.xlabel("epoch")
        ax14.set_xscale('log')
        fmt = '%.2f%%'  # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax14.yaxis.set_major_formatter(xticks)
        pp.savefig(fig)
        pp.close()
        plt.close()


