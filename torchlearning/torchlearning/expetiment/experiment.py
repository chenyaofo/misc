import os
import time
import uuid


class Experiment(object):
    def __init__(self, save_path=None):
        self.id = uuid.uuid4()
        if save_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.save_path = os.path.join("experiments", timestamp)
            os.makedirs(self.save_path, exist_ok=False)
        else:
            self.save_path = save_path


