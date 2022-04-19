from collections import defaultdict
import numbers
import json


class TrainRecorder(dict):
    def __init__(self, **kwargs):
        super(TrainRecorder, self).__init__()
        self.update(kwargs)

    def set_phase(self,phase):
        self.phase = phase
        if self.get(self.phase) is None:
            self[self.phase] = defaultdict(list)

    def train(self):
        self.set_phase("engine")

    def eval(self):
        self.set_phase("validation")

    def test(self):
        self.set_phase("test")

    def add_scalar_summary(self, **kwargs):
        for key,value in kwargs.items():
            self[self.phase][key].append(value)

    def __getattr__(self, attr):
        return self[attr]

    @property
    def json(self):
        return json.dumps(self, indent=2, )

    def dumps(self, filename):
        with open(filename, "w") as f:
            json.dump(self, f)

    @staticmethod
    def loads(filename):
        recorder = TrainRecorder()
        with open(filename, "r") as f:
            recorder.update(json.load(f))
        return recorder

if __name__ == '__main__':
    r = TrainRecorder(name="ResNet")
    r.train()
    r.add_scalar_summary(loss=0.1)
    d = dict(loss=0.01)
    r.add_scalar_summary(**d)
    print(r.json)
    r.dumps("/tmp/d")
    r1 = TrainRecorder.loads("/tmp/d")
    print(r1.json)
