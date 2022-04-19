class LearningContext(object):
    def __init__(self, phase, epoch, net, device, loader, criterion, optimizer=None, scheduler=None, recorder=None,
                 logger=None):
        self.phase = phase
        self.epoch = epoch
        self.net = net
        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = recorder
        self.logger = logger



