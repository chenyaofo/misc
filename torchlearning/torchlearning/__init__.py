from .config import conf
from .logging import logger
import pyhocon
import logging

__all__ = [
    "conf",
    "logger",
]


# @property
# def config() -> pyhocon.ConfigTree:
#     pass
#
#
# @property
# def logger() -> logging.logger:
#     pass
