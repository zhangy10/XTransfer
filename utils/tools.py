"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
from difflib import SequenceMatcher
import PIL
import torch
from PIL import Image
from six.moves import cPickle as pickle
import psutil

__all__ = [
    "mkdir_if_missing",
    "check_isfile",
    "read_json",
    "write_json",
    "set_random_seed",
    "download_url",
    "read_image",
    "collect_env_info",
    "listdir_nohidden",
    "get_most_similar_str_to_a_from_b",
    "check_availability",
    "tolist_if_not",
    "save_dict",
    "load_dict"
]


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def download_url(url, dst):
    """Download file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    return Image.open(path).convert("RGB")


def collect_env_info():
    """Return env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen


def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "The requested one is expected "
            "to belong to {}, but got [{}] "
            "(do you mean [{}]?)".format(available, requested, psb_ans)
        )


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x

def get_cpu_mem():
    pid = os.getpid()
    p = psutil.Process(pid)
    cpu_percent = p.cpu_percent()
    mem_percent = p.memory_percent()
    mem_rss = p.memory_info()
    return cpu_percent, mem_percent, mem_rss[0]


import torch
import numpy as np


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)


def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))

    return np.mean(cl_sparsity)

# if __name__ =="__main__":
#     while True:
#         a, b , c = get_cpu_mem()
#         print(a , b , c)
