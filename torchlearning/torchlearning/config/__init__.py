import sys
import argparse
import pyhocon

__all__ = [
    "conf"
]

parser = argparse.ArgumentParser(description="train a network")
parser.add_argument("-c", "--config", type=str, nargs='?', help="the path to config file.", default=None,
                    required=False)
args, remain = parser.parse_known_args(sys.argv)

if args.config is None:
    conf = None
else:
    conf = pyhocon.ConfigFactory.parse_file(args.config)
