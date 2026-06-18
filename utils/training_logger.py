from collections import defaultdict
from .tools import load_dict, save_dict
from .tools import mkdir_if_missing
import os
import numpy as np

__all__ = [
    "TorchLogger",
    "RSRLogger"
]


class TorchLogger:
    """Write training logging into pickle dictionary.

    Args:
        fpath (str): directory to save logging file.

    """

    def __init__(self, fpath=None):
        mkdir_if_missing(fpath)
        self.save_path = os.path.join(fpath, 'log_dict.pkl')
        self.keys = ['train_accuracy', 'train_loss', 'epoch_time', 'test_time', 'eposide_time', 'support_time',
                     'query_time', 'val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'train_cpu', 'train_mem',
                     'train_mem_rss', "flops", "params", 'source_1_train_accuracy', 'source_2_train_accuracy',
                     'source_3_train_accuracy', 'source_4_train_accuracy', 'source_5_train_accuracy',
                     'source_1_test_accuracy', 'source_2_test_accuracy', 'source_3_test_accuracy',
                     'source_4_test_accuracy', 'source_5_test_accuracy', 'source_1_loss', 'source_2_loss',
                     'source_3_loss', 'source_4_loss', 'source_5_loss', 's1_train_time', 's2_train_time',
                     'distill_time', 'gen_loss', 'disc_loss', 'source_1_encoder_epoch_time',
                     'source_2_encoder_epoch_time',
                     'source_3_encoder_epoch_time', 'source_4_encoder_epoch_time', 'source_5_encoder_epoch_time',
                     'source_1_classifier_epoch_time', 'source_2_classifier_epoch_time',
                     'source_3_classifier_epoch_time',
                     'source_4_classifier_epoch_time', 'source_5_classifier_epoch_time', 'before_train_accuracy',
                     'prune_train_accuracy', 'before_test_accuracy', 'prune_test_accuracy',
                     'source_1_gen_loss', 'source_2_gen_loss', 'source_3_gen_loss', 'source_4_gen_loss',
                     'source_5_gen_loss',
                     'source_1_dis_loss', 'source_2_dis_loss', 'source_3_dis_loss', 'source_4_dis_loss',
                     'source_5_dis_loss', 'backbone_structure']
        self.dic = defaultdict(list)

    def __exit__(self, *args):
        self.close()

    def write(self, key, value):
        if self.check_key(key):
            self.dic[key].append(value)
        else:
            raise KeyError("This key is not supported yet!")

    def close(self):
        for key, item in self.dic.items():
            self.dic[key] = np.asarray(item)
        self.save()
        print('Save training log into {}.'.format(self.save_path))

    def get_history(self, key):
        return self.dic[key]

    def check_key(self, key):
        if key in self.keys:
            return True
        else:
            return False

    def save(self):
        save_dict(self.dic, self.save_path)


class RSRLogger:
    """Write training logging into pickle dictionary.

    Args:
        fpath (str): directory to save logging file.

    """

    def __init__(self, fpath=None):
        mkdir_if_missing(fpath)
        self.save_path = os.path.join(fpath, 'log_dict.pkl')
        self.keys = ['before_train_acc', 'train_acc', 'prune_train_acc', 'before_test_acc', 'test_acc',
                     'prune_test_acc', 'before_train_pca_acc', 'train_pca_acc', 'prune_train_pca_acc',
                     'before_test_pca_acc', 'test_pca_acc', 'prune_test_pca_acc', 'train_pt_acc', 'test_pt_acc',
                     'prune_train_pt_acc', 'prune_test_pt_acc', 'backbone_structure',
                     'macs', 'params', 'time', 'layer_time', 'users', 'finetune_knn_train_acc', 'finetune_knn_test_acc',
                     'finetune_linear_train_acc', 'finetune_linear_test_acc', 'before_distance', 'after_distance',
                     'epoch_loss', 'estimate_score', 'score']
        self.dic = defaultdict(list)

    def __exit__(self, *args):
        self.close()

    def write(self, key, value):
        if self.check_key(key):
            self.dic[key].append(value)
        else:
            raise KeyError("This key is not supported yet!")

    def close(self):
        for key, item in self.dic.items():
            self.dic[key] = item
        self.save()
        print('Save training log into {}.'.format(self.save_path))

    def get_history(self, key):
        return self.dic[key]

    def check_key(self, key):
        if key in self.keys:
            return True
        else:
            return False

    def save(self):
        save_dict(self.dic, self.save_path)

