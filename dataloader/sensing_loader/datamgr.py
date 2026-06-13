import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
import os

from dataloader.dataset import SimpleDataset, Datum
from dataloader.transforms import Raw1D, Raw2D, Resize, GADF
from utils import load_dict
from torch.utils.data.sampler import RandomSampler

DATAROOT = 'E:/Datasets/Source/Sensing/'


class DataManager:
    def get_user_split_loader(self, batch_size, trans_method='raw', train_users=[], test_users=[], num_workers=0):
        self.trans_method = trans_method
        transform = self.build_transform()
        train = self.load_user_data(train_users)
        test = self.load_user_data(test_users)

        train_loader = torch.utils.data.DataLoader(
            SimpleDataset(train, transform),
            batch_size=batch_size,
            sampler=RandomSampler(train),
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            SimpleDataset(test, transform),
            batch_size=batch_size,
            sampler=RandomSampler(test),
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, test_loader

    def get_percentage_split_loader(self, batch_size, trans_method='raw', test_split=0.2, num_workers=0):
        self.trans_method = trans_method
        transform = self.build_transform()

        paths, labels, users = self.load_user_data_list()
        path_train, path_test, y_train, y_test = train_test_split(paths, labels, test_size=test_split, stratify=users)

        train = []
        for path, label in zip(path_train, y_train):
            train.append(Datum(path, label))
        test = []
        for path, label in zip(path_test, y_test):
            test.append(Datum(path, label))

        train_loader = torch.utils.data.DataLoader(
            SimpleDataset(train, transform),
            batch_size=batch_size,
            sampler=RandomSampler(train),
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            SimpleDataset(test, transform),
            batch_size=batch_size,
            sampler=RandomSampler(test),
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, test_loader

    def load_user_data(self, users):
        items = []
        meta_data = load_dict(self.data_file)
        for u, d in meta_data.items():
            if u in users:
                for label, path_list in d.items():
                    for path in path_list:
                        items.append(Datum(os.path.join(self.data_root, path), label, u))
        return items

    def load_user_data_list(self):
        meta_data = load_dict(self.data_file)
        paths = []
        labels = []
        users = []
        for u, d in meta_data.items():
            for label, path_list in d.items():
                for path in path_list:
                    paths.append(os.path.join(self.data_root, path))
                    labels.append(label)
                    users.append(u)
        return paths, labels, users

    def build_transform(self):
        # build transformer accordingly
        self.trans_loader = TransformLoader(self.mean, self.std)
        transform = self.trans_loader.get_composed_transform(self.trans_method)
        return transform


class TransformLoader:
    def __init__(self, mean=[], std=[]):
        self.mean = mean
        self.std = std
        self.image_size = 224

    def get_composed_transform(self, method='raw', **kwargs):
        if method == 'raw':
            transform = transforms.Compose([Raw1D()])
        elif method == 'Raw2D':
            transform = transforms.Compose([Raw2D(0), Resize(self.image_size)])
        elif method == 'gadf':
            transform = transforms.Compose([GADF(self.image_size), Resize(self.image_size)])
        else:
            raise ValueError('No {} method!!'.format(method))
        return transform


class MHEALTH(DataManager):
    n_way = 11
    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_channels = 23
    all_users = [str(i) for i in range(1, 11)]
    mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]
    data_root = os.path.join(DATAROOT, 'MHEALTH')
    data_file = 'filelists/MHEALTH/MHEALTH.pkl'

    def __init__(self, data_root, data_file, **kwargs):
        super(MHEALTH, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class OPPORTUNITY(DataManager):
    n_way = 11
    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_channels = 77
    all_users = [str(i) for i in range(1, 2)]
    # mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    # std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]

    mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    data_root = os.path.join(DATAROOT, 'OPPORTUNITY')
    data_file = 'filelists/OPPORTUNITY/OPPORTUNITY.pkl'

    def __init__(self, data_root, data_file, **kwargs):
        super(OPPORTUNITY, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class PAMAP2(DataManager):
    n_way = 7
    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_channels = 9
    all_users = [str(i) for i in range(1, 2)]
    mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]
    data_root = os.path.join(DATAROOT, 'PAMAP2')
    data_file = 'filelists/PAMAP2/PAMAP2.pkl'

    def __init__(self, data_root, data_file, **kwargs):
        super(PAMAP2, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class EMG(DataManager):
    n_way = 6
    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_channels = 8
    all_users = [str(i) for i in range(1, 37)]
    mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]
    data_root = os.path.join(DATAROOT, 'sEMG')
    data_file = 'filelists/sEMG/sEMG.pkl'

    def __init__(self, data_root, data_file, **kwargs):
        super(EMG, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class UniMiB(DataManager):
    n_way = 8
    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_channels = 1
    all_users = [str(i) for i in range(1, 2)]
    mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]

    # data_root = os.path.join(DATAROOT, 'UniMiB')
    # data_file = 'filelists/UniMiB/UniMiB.pkl'

    def __init__(self, data_root, data_file, **kwargs):
        super(UniMiB, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


def sensing_dataloader(dataset='MHEALTH'):
    if dataset == 'MHEALTH':
        return MHEALTH
    elif dataset == 'OPPORTUNITY':
        return OPPORTUNITY
    elif dataset == 'PAMAP2':
        return PAMAP2
    elif dataset == 'sEMG':
        return EMG
    elif dataset == 'UniMiB':
        return UniMiB
    else:
        raise ValueError("It is not supported for {}.".format(dataset))


if __name__ == "__main__":
    n_eposide = 1
    trans_method = 'raw'
    dataset = 'UniMiB'
    data_obj = sensing_dataloader(dataset)
    dataloader = data_obj()
    train, test = dataloader.get_percentage_split_loader(batch_size=64, trans_method='raw', num_workers=0)

    print(train.__len__())
    for x, y in train:
        print(x.shape, y.shape)

    print(test.__len__())
    for x, y in test:
        print(x.shape, y.shape)
