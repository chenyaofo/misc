import os
import sys
import time
import logging

experiment_path = os.environ.get("experiment_path")
if experiment_path is None:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    experiment_path = os.path.join("experiments", timestamp)
os.makedirs(experiment_path, exist_ok=False)

logger = logging.getLogger("torchlearning")

formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s: %(message)s"
)

file_handler = logging.FileHandler(os.path.join(experiment_path, "training.log"))
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)

__all__ = [
    "logger",
    "experiment_path"
]