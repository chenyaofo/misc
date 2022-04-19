from abc import abstractmethod

from ..engine import Engine

class Plugin(object):
    def __init__(self):
        pass

    @abstractmethod
    def attach(self,engine:Engine):
        raise NotImplementedError()