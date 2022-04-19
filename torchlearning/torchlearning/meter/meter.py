class Meter(object):
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def attach(self,engine):
        raise NotImplementedError()

    @property
    def value(self):
        return None

    @property
    def record(self):
        return None

