from .group import GroupResize, GroupRandomCrop, GroupCenterCrop, GroupToTensor, GroupRandomHorizontalFlip

__all__ = [
    "GroupToTensor",
    "GroupRandomHorizontalFlip",
    "GroupCenterCrop",
    "GroupRandomCrop",
    "GroupResize",
    "Bytes2Image"
]

from . import functional as F


class Bytes2Image(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, bytes):
        return F.bytes2image(bytes, self.mode)
