import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from dataloader.target_loader.datamgr import target_dataloader
from modeling import Header
from utils import load_dict
from dataloader.transforms import *
from xtransfer.paths import get_target_paths
from dataloader.transforms import Resize


class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.labels = label
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


class LoaderWrapper:
    def __init__(self, dataname, dataroot, metafile, resize, trans_method, n_support, n_query, mode, users=None,
                 n_eposide=15, seed=None, afterheader=False, num_workers=0, return_validation=False, **kwargs):
        dataset_class = target_dataloader(dataname)
        self.users = dataset_class.all_users
        self.dataset = dataset_class(dataroot, metafile)
        self.resize = resize
        self.trans_method = trans_method
        self.n_support = n_support
        self.n_query = n_query
        self.mode = mode
        self.n_eposide = n_eposide
        self.afterheader = afterheader
        self.num_workers = num_workers
        self.regression = kwargs['regression']
        self.sbp = kwargs['sbp']
        if users is not None:
            self.users = users
        self.seed = seed
        self.return_validation = return_validation
        self.n_val = 5
        if self.return_validation:
            self.n_query += self.n_val

        if afterheader:
            if trans_method == 'Raw2D':
                n_channels = 1
            else:
                n_channels = self.dataset.num_channels
            self.header = Header(num_channels=n_channels)

    def get_data_size(self, data):
        if len(data.size()) > 4:
            _, _, c, h, w = data.size()
            return [c, h, w]
        else:
            _, _, c, w = data.size()
            return [c, w]

    def return_loaders(self, epo_idx=0):
        n_way = self.dataset.n_way
        self.loader = self.dataset.get_fewshot_loader(self.resize, self.trans_method, self.n_support, self.n_query,
                                                      self.mode, self.n_eposide, seed=self.seed, users=self.users)
        if self.return_validation:
            n_query = self.n_query - self.n_val
        else:
            n_query = self.n_query
        for i, (x, y) in enumerate(self.loader):
            if i == epo_idx:
                data = x
                if self.afterheader:
                    data = self.header(data)
                    # data = data.detach()
                target = y
                break
        size_list = self.get_data_size(data)
        train_x = data[:, :self.n_support].contiguous().view(n_way * self.n_support, *size_list)
        train_y = target[:, :self.n_support].contiguous().view(n_way * self.n_support)

        val_x = data[:, self.n_support:self.n_support + self.n_val].contiguous().view(n_way * self.n_val, *size_list)
        val_y = target[:, self.n_support:self.n_support + self.n_val].contiguous().view(n_way * self.n_val)

        test_x = data[:, -n_query:].contiguous().view(n_way * n_query, *size_list)
        test_y = target[:, -n_query:].contiguous().view(n_way * n_query)

        train_dataset = CustomDataset(train_x, train_y)
        val_dataset = CustomDataset(val_x, val_y)
        test_dataset = CustomDataset(test_x, test_y)
        train = DataLoader(train_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=train_x.size(0),
                           shuffle=False)
        val = DataLoader(val_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=val_x.size(0),
                         shuffle=False)
        test = DataLoader(test_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=test_x.size(0),
                          shuffle=False)
        return train, val, test

    def return_regression_loader(self, epo_idx=0):
        resizer = Resize(224)

        n_way = self.dataset.n_way
        self.loader = self.dataset.get_fewshot_loader(self.resize, self.trans_method, self.n_support, self.n_query,
                                                      self.mode, self.n_eposide, seed=self.seed, users=self.users,
                                                      regression=self.regression)
        if self.return_validation:
            n_query = self.n_query - self.n_val
        else:
            n_query = self.n_query
        for i, (x, y) in enumerate(self.loader):
            if i == epo_idx:
                data = x
                if self.afterheader:
                    data = self.header(data)
                    # data = data.detach()
                target = y
                break
        size_list = self.get_data_size(data)
        train = data[:, :self.n_support].contiguous().view(n_way * self.n_support, *size_list)

        data_length = train.size(2) - 2
        if not self.sbp:
            label_idx = data_length + 1
        else:
            label_idx = data_length

        train_x, train_y = train[:, :, :data_length], train[:, 0, label_idx]

        # random get val data index:
        idx = np.arange(self.n_query)
        bins = np.array_split(idx, self.n_val)
        val_indices = np.empty(self.n_val, dtype=int)
        for i, bin_data in enumerate(bins):
            random_index = np.random.choice(bin_data, 1, replace=False)
            val_indices[i] = random_index
        tdata = data[:, self.n_support:].contiguous().view(n_way * self.n_query, *size_list)

        val = tdata[val_indices].contiguous().view(n_way * self.n_val, *size_list)
        val_x, val_y = val[:, :, :data_length], val[:, 0, label_idx]

        mask = np.ones(tdata.size(0), dtype=bool)
        mask[val_indices] = False
        test = tdata[mask].contiguous().view(n_way * n_query, *size_list)
        test_x, test_y = test[:, :, :data_length], test[:, 0, label_idx]

        # train_x = resizer(train_x).unsqueeze(1)
        # val_x = resizer(val_x).unsqueeze(1)
        # test_x = resizer(test_x).unsqueeze(1)

        train_dataset = CustomDataset(train_x, train_y)
        val_dataset = CustomDataset(val_x, val_y)
        test_dataset = CustomDataset(test_x, test_y)

        train = DataLoader(train_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=train_x.size(0),
                           shuffle=False)
        val = DataLoader(val_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=val_x.size(0),
                         shuffle=False)
        test = DataLoader(test_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=test_x.size(0),
                          shuffle=False)
        return train, val, test

    def get_dataset(self):
        return self.dataset


def create_dataloader(data_name, resize, n_shot, n_query=15, trans_method='stft', mode='d', epo_idx=0,
                      afterheader=False, seed=None, num_workers=0, return_validation=False, **kwargs):
    select_users = kwargs['users']
    if data_name == 'OXiod':
        train, test = get_oxiod_few_data(resize)
        return train, test
    else:
        dataroot, metafile = get_target_paths(data_name)

    regression = kwargs['regression']
    sbp = kwargs['sbp']

    wrapper = LoaderWrapper(data_name, dataroot, metafile, resize, trans_method, n_shot, n_query, mode, select_users,
                            seed=seed, afterheader=afterheader, num_workers=num_workers,
                            return_validation=return_validation, regression=regression, sbp=sbp)
    if not regression:
        train, val, test = wrapper.return_loaders(epo_idx)
    else:
        train, val, test = wrapper.return_regression_loader(epo_idx)
    users = wrapper.loader.batch_sampler.sampled_users[-1]
    if return_validation:
        return train, val, test, users
    return train, test, users


def get_oxiod_few_data(resize):
    nperseg = 22
    datafile = 'E:/GitHub/6-DOF-Inertial-Odometry/outputs/few_data.pkl'
    data = load_dict(datafile)
    x_gyro = np.float32(data['x_gyro'])
    x_acc = np.float32(data['x_acc'])
    y_delta_p = np.float32(data['y_delta_p'])
    y_delta_q = np.float32(data['y_delta_q'])

    transform = transforms.Compose(
        [STFT(nperseg), Resize(resize)])

    x = np.concatenate([x_gyro, x_acc], axis=1)
    y_train = torch.from_numpy(np.arange(0, len(x)))
    y = torch.from_numpy(np.concatenate([y_delta_p, y_delta_q], axis=1))
    train_dataset = CustomDataset(x, y_train, transform=transform)
    test_dataset = CustomDataset(x, y, transform=transform)

    train = DataLoader(train_dataset, num_workers=0, pin_memory=True, batch_size=x.shape[0],
                       shuffle=False)
    test = DataLoader(test_dataset, num_workers=0, pin_memory=True, batch_size=x.shape[0],
                      shuffle=False)

    return train, test

