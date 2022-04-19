from .split import Split
from .reader import MioReader
from .writer import MioWriter


__version__ = "v0.1.0"

MIO = MioReader

__all__ = [
    "Split",
    "MIO",
    "MioReader",
    "MioWriter"
]
