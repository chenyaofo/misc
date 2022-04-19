from math import ceil
import pathlib
import hashlib
from collections import OrderedDict
import socket
import tempfile
import torch


def chunks(arr, m):
    n = int(ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def get_file_sha256(filename):
    m = hashlib.sha256()


class FileHasher(object):
    def __init__(self, algorithm="md5"):
        if not hasattr(hashlib, algorithm):
            raise Exception("Not support such algorithm().".format(algorithm))
        self.algorithm = getattr(hashlib, algorithm)

    def __call__(self, filename):
        hasher = self.algorithm()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @property
    def algorithms_available(self):
        return hashlib.algorithms_available

# follow the statement on
# https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url
def save_model_with_hash(model: torch.nn.Module, path):
    if isinstance(model,torch.nn.Module):
        state_dict = model.cpu().state_dict()
    else:
        state_dict = model

    dst = pathlib.Path(path)
    tmp_pt = dst.with_name("temporary").with_suffix(".pt")
    torch.save(state_dict, tmp_pt)
    sha1_hash = FileHasher("sha256")
    signature = sha1_hash(tmp_pt)[:8]
    purename, extension = dst.stem, dst.suffix
    dst = dst.with_name("{}-{}".format(purename, signature)).with_suffix(extension)
    tmp_pt.rename(dst)


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

if __name__ == '__main__':
    import torchvision.models

    net = torchvision.models.resnet18(pretrained=True)
    save_model_with_hash(net, "/tmp/resnet.pt")
