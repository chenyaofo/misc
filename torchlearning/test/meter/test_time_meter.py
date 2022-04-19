import time
import math
from torchlearning.meter import TimeMeter


class TestTimeMeter(object):
    def test_time(self):
        m = TimeMeter()
        time.sleep(3)
        assert math.isclose(3.0, round(m.value))
