class LargerMetric(object):
    def __init__(self, initial_value=float("-inf"), attribute_map=lambda x: x):
        self.current_value = initial_value
        self.attribute_map=attribute_map

    def __call__(self, ctx):
        if self.attribute_map(ctx) > self.current_value:
            self.current_value = self.attribute_map(ctx)
            return True
        else:
            return False


class SmallerMetric(object):
    def __init__(self, initial_value=float("inf"), attribute_map=lambda x: x):
        self.current_value = initial_value
        self.attribute_map = attribute_map

    def __call__(self, ctx):
        if self.attribute_map(ctx) < self.current_value:
            self.current_value = self.attribute_map(ctx)
            return True
        else:
            return False
