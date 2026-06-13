import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
import os

from dataloader.dataset import UserSetDataset, EpisodicBatchSampler, EpisodicBatchSamplerMeta, SimpleDataset, Datum
from dataloader.transforms import STFT, GADF, Resize, Raw1D, Raw2Image, Raw2D, ToTensor, Raw2D_square, Resize_1D
from utils import load_dict
from torch.utils.data.sampler import RandomSampler


class DataManager:
    all_mode = ['p', 'i', 'd']  # personalized, independent, dependent
    is_img = False

    def get_fewshot_loader(self, resize=224, trans_method='stft', n_support=5, n_query=15, mode='d', n_eposide=10,
                           users=None, num_workers=0, seed=None, percentage=1.0, regression=False, **kwargs):
        self.resize = resize
        self.trans_method = trans_method
        self.n_support = n_support
        self.n_query = n_query
        self.mode = mode
        self.n_eposide = n_eposide
        self.percentage = percentage
        self.regression = regression
        if seed is None:
            self.seed = [i for i in range(n_eposide)]
        else:
            self.seed = [(seed + i) for i in range(n_eposide)]
        if users is not None:
            self.users = users
        else:
            self.users = self.all_users

        transform = self.build_transform()

        self.dataset = UserSetDataset(data_file=self.data_file, data_root=self.data_root, users=self.users,
                                      n_way=self.n_way, n_shot=self.n_support, n_query=self.n_query,
                                      regression=self.regression,
                                      transform=transform, mode=self.mode, load_img=self.is_img, percentage=percentage)
        self.sampler = EpisodicBatchSampler(len(self.dataset), self.n_way, self.n_eposide, self.seed,
                                            self.dataset.samplers, self.mode)
        data_loader_params = dict(batch_sampler=self.sampler, num_workers=num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self.dataset, **data_loader_params)
        return data_loader

    # meta mode:
    # first half tasks of epoch is in maml mode and second half tasks is in meta mode
    # metasense n_epoch = 10, we set n_eposide = n_epoch * 2 * num_source_users
    # maml mode:
    # all tasks of epoch is in maml mode
    # maml n_epoch = 10, we set n_eposide = n_epoch * num_source_users
    def get_meta_fewshot_loader(self, resize=224, trans_method='stft', n_support=5, n_query=15, n_eposide=10,
                                users=None, num_workers=0, seed=None, mode='meta', percentage=1.0, **kwargs):
        self.resize = resize
        self.trans_method = trans_method
        self.n_support = n_support
        self.n_query = n_query
        self.n_eposide = n_eposide
        self.percentage = percentage
        if seed is None:
            self.seed = [i for i in range(n_eposide)]
        else:
            self.seed = [(seed + i) for i in range(n_eposide)]
        if users is not None:
            self.users = users
        else:
            self.users = self.all_users

        transform = self.build_transform()

        self.dataset = UserSetDataset(data_file=self.data_file, data_root=self.data_root, users=self.users,
                                      n_way=self.n_way, n_shot=self.n_support, n_query=self.n_query,
                                      transform=transform, mode='p', load_img=self.is_img, percentage=percentage)
        self.sampler = EpisodicBatchSamplerMeta(len(self.dataset), self.n_way, self.n_eposide, self.seed,
                                                self.dataset.samplers, mode, user_list=users)
        data_loader_params = dict(batch_sampler=self.sampler, num_workers=num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self.dataset, **data_loader_params)
        return data_loader

    def get_user_split_loader(self, batch_size, resize=224, trans_method='', train_users=[], test_users=[],
                              num_workers=0):
        self.resize = resize
        self.trans_method = trans_method
        transform = self.build_transform()
        train = self.load_user_data(train_users)
        test = self.load_user_data(test_users)

        train_loader = torch.utils.data.DataLoader(
            SimpleDataset(train, transform, is_img=self.is_img),
            batch_size=batch_size,
            sampler=RandomSampler(train),
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            SimpleDataset(test, transform, is_img=self.is_img),
            batch_size=batch_size,
            sampler=RandomSampler(test),
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, test_loader

    def get_percentage_split_loader(self, batch_size, resize=224, trans_method='', test_split=0.2, num_workers=0):
        self.resize = resize
        self.trans_method = trans_method
        transform = self.build_transform()

        paths, labels, users = self.load_user_data_list()
        if test_split == 0:
            path_train, y_train = paths, labels
            path_test, y_test = paths, labels
        else:
            path_train, path_test, y_train, y_test = train_test_split(paths, labels, test_size=test_split,
                                                                      stratify=users)

        train = []
        for path, label in zip(path_train, y_train):
            train.append(Datum(path, label))
        test = []
        for path, label in zip(path_test, y_test):
            test.append(Datum(path, label))

        train_loader = torch.utils.data.DataLoader(
            SimpleDataset(train, transform, is_img=self.is_img),
            batch_size=batch_size,
            sampler=RandomSampler(train),
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            SimpleDataset(test, transform, is_img=self.is_img),
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
        self.trans_loader = TransformLoader(self.resize, self.nperseg)
        if self.trans_method == 'stft':
            mean, std = self.stft_mean, self.stft_std
        elif self.trans_method == 'gadf':
            mean, std = self.gadf_mean, self.gadf_std
        elif self.trans_method == 'raw2img':
            mean, std = self.mean, self.std
        elif self.trans_method == 'Raw1D':
            mean, std = [], []
        else:
            mean, std = [], []
        transform = self.trans_loader.get_composed_transform(self.trans_method, mean, std)
        return transform


class TransformLoader:
    def __init__(self, image_size, nperseg=30):
        self.image_size = image_size
        self.nperseg = nperseg

    def get_composed_transform(self, method='stft', mean=[], std=[], **kwargs):
        if method == 'stft':
            # transform = transforms.Compose(
            #     [STFT(self.nperseg), Resize(self.image_size), transforms.Normalize(mean, std)])
            transform = transforms.Compose(
                [STFT(self.nperseg), Resize(self.image_size)])
        elif method == 'stft_raw':
            # transform = transforms.Compose(
            #     [STFT(self.nperseg), Resize(self.image_size), transforms.Normalize(mean, std)])
            transform = transforms.Compose(
                [STFT(self.nperseg), ToTensor(), ])
        elif method == 'gadf':
            # transform = transforms.Compose(
            #     [GADF(self.image_size), Resize(self.image_size), transforms.Normalize(mean, std)])
            transform = transforms.Compose(
                [GADF(self.image_size), Resize(self.image_size)])
        elif method == 'img':
            transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        elif method == 'raw2img':
            transform = transforms.Compose(
                [Raw2Image(), transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        elif method == 'Raw2D':
            transform = transforms.Compose(
                [Raw2D(0), Resize(self.image_size)])
        elif method == 'Raw2D_square':
            transform = transforms.Compose(
                [Raw2D_square(0), Resize(self.image_size)])
        elif method == 'Raw':
            transform = transforms.Compose(
                [Raw2D(1)])
        elif method == 'Raw1D':
            transform = transforms.Compose([Raw1D()])
        elif method == 'Raw1D_resize':
            transform = transforms.Compose([Resize_1D(self.image_size)])
        else:
            raise ValueError('No {} method!!'.format(method))
        return transform


class HHAR(DataManager):
    n_way = 6
    nperseg = 22
    stft_mean = [0.0735, 0.0573, 0.0516, 0.0609, 0.0502, 0.0571]
    stft_std = [0.1695, 0.1294, 0.1187, 0.1393, 0.1158, 0.1299]
    num_channels = 6
    all_users = [str(i) for i in range(0, 9)]
    # nperseg = 30
    # stft_mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    # stft_std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]
    gadf_mean = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    gadf_std = [0.5311, 0.5214, 0.5349, 0.5735, 0.5504, 0.5798]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, data_root, data_file, **kwargs):
        super(HHAR, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class WESAD(DataManager):
    n_way = 3
    # nperseg = 50  #(26,53)
    nperseg = 70  # (36,38)
    num_channels = 10
    all_users = [str(i) for i in range(0, 15)]
    stft_mean = [30.0023]
    stft_std = [94.4520]
    gadf_mean = [0.0000]
    gadf_std = [0.4239]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, data_root, data_file, **kwargs):
        super(WESAD, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class HHARDeepSense(DataManager):
    n_way = 6
    nperseg = 30
    num_channels = 20
    all_users = [str(i) for i in range(0, 9)]
    stft_mean = [0.0562, 0.0418, 0.0380, 0.0454, 0.0370, 0.0426]
    stft_std = [0.1503, 0.1098, 0.1016, 0.1216, 0.0998, 0.1134]
    gadf_mean = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    gadf_std = [0.5311, 0.5214, 0.5349, 0.5735, 0.5504, 0.5798]

    def __init__(self, data_root, data_file, **kwargs):
        super(HHARDeepSense, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class Face(DataManager):
    n_way = 7
    class_list = ['Neutral', 'Happiness', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    num_channels = 3
    mean = [0.0778, 0.3081, 0.8488]
    std = [0.2113, 0.3341, 0.2296]
    all_users = [str(i) for i in range(0, 5)]
    is_img = True

    def __init__(self, data_root, data_file, **kwargs):
        super(Face, self).__init__()
        self.data_root = data_root
        self.data_file = data_file

    def build_transform(self):
        # build transformer accordingly
        self.trans_loader = TransformLoader(self.resize, 0)
        transform = self.trans_loader.get_composed_transform(method='img', mean=self.mean, std=self.std)
        return transform


class ChestX(DataManager):
    n_way = 5
    num_channels = 3
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    all_users = [str(i) for i in range(0, 1)]
    is_img = True

    def __init__(self, data_root, data_file, **kwargs):
        super(ChestX, self).__init__()
        self.data_root = data_root
        self.data_file = data_file

    def build_transform(self):
        # build transformer accordingly
        self.trans_loader = TransformLoader(self.resize, 0)
        transform = self.trans_loader.get_composed_transform(method='img', mean=self.mean, std=self.std)
        return transform


class Finger(DataManager):
    n_way = 8
    # nperseg = 5   # (11,11)
    # nperseg = 100  # (11,11)
    nperseg = 20  # (11,11)
    num_channels = 7
    all_users = [str(i) for i in range(0, 10)]
    # all_users = ['10']
    stft_mean = [158.0829, 166.1225, 599.8087, 67.8330, 43.7092, 36.5722]
    stft_std = [324.9713, 328.2784, 1064.5900, 176.9834, 135.8807, 97.8635]
    gadf_mean = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    gadf_std = [0.5311, 0.5214, 0.5349, 0.5735, 0.5504, 0.5798]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, data_root, data_file, **kwargs):
        super(Finger, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class Ultrasound(DataManager):
    n_way = 10
    # nperseg = 20 #(11,14)
    nperseg = 22  # (12,13)
    num_channels = 1
    all_users = [str(i) for i in range(0, 10)]
    stft_mean = [0.0597]
    stft_std = [0.1678]
    gadf_mean = [0.0000]
    gadf_std = [0.6526]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, data_root, data_file, **kwargs):
        super(Ultrasound, self).__init__()
        self.data_root = data_root
        self.data_file = data_file


class BloodPressure(DataManager):
    n_way = 1
    num_channels = 1
    # all_users = [str(i) for i in range(0, 10)]
    all_users = ['0', '1', '2', '5', '7', '8', '9']
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, data_root, data_file, **kwargs):
        super(BloodPressure, self).__init__()
        self.data_root = data_root
        self.data_file = data_file

    def get_fewshot_loader(self, resize=224, trans_method='stft', n_support=5, n_query=15, mode='d', n_eposide=10,
                           users=None, num_workers=0, seed=None, percentage=1.0, regression=False, **kwargs):
        self.resize = resize
        self.trans_method = trans_method
        self.n_support = n_support
        self.n_query = n_query
        self.mode = mode
        self.n_eposide = n_eposide
        self.percentage = percentage
        self.regression = regression
        if seed is None:
            self.seed = [i for i in range(n_eposide)]
        else:
            self.seed = [(seed + i) for i in range(n_eposide)]
        if users is not None:
            self.users = users
        else:
            self.users = self.all_users

        transform = self.build_transform()

        self.dataset = UserSetDataset(data_file=self.data_file, data_root=self.data_root, users=self.users,
                                      n_way=self.n_way, n_shot=self.n_support, n_query=self.n_query, regression=True,
                                      transform=transform, mode=self.mode, load_img=self.is_img, percentage=percentage)
        self.sampler = EpisodicBatchSampler(len(self.dataset), self.n_way, self.n_eposide, self.seed,
                                            self.dataset.samplers, self.mode)
        data_loader_params = dict(batch_sampler=self.sampler, num_workers=num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self.dataset, **data_loader_params)
        return data_loader

    def build_transform(self):
        # build transformer accordingly
        self.trans_loader = TransformLoader(self.resize, 0)
        transform = self.trans_loader.get_composed_transform(method='Raw1D', mean=self.mean, std=self.std)
        return transform


def target_dataloader(dataset='HHAR'):
    if dataset == 'HHAR':
        return HHAR
    elif dataset == 'Face':
        return Face
    elif dataset == 'HHARDeepSense':
        return HHARDeepSense
    elif dataset == 'WESAD':
        return WESAD
    elif dataset == 'Finger':
        return Finger
    elif dataset == 'Ultrasound':
        return Ultrasound
    elif dataset == 'ChestX':
        return ChestX
    elif dataset == 'BloodPressure':
        return BloodPressure
