# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import json
import numpy as np
import math
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from utils import load_dict, set_random_seed, check_isfile
from dataloader.transforms import Target


def load_file(path):
    data = None
    if '.jpg' in path:
        data = Image.open(path).convert('RGB')
    elif path.endswith('.png'):
        data = Image.open(path).convert('RGB')
    elif '.pkl' in path:
        data = load_dict(path)
    elif '.npy' in path:
        data = np.load(path)
    else:
        raise ValueError('This format of file ({}) is not supported!'.format(os.path.split(path))[-1])
    return data


class SimpleDataset(TorchDataset):
    def __init__(self, data_source, transform=None, target_transform=None, is_img=False):
        self.data_source = data_source
        self.transform = transform
        if target_transform is None:
            self.target_transform = transforms.Compose([Target(), ])
        else:
            self.target_transform = target_transform
        self.is_img = is_img

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        path = item.path
        target = item.label
        # if self.is_img:
        #     data = Image.open(path).convert('RGB')
        # else:
        #     data = load_dict(path)

        data = load_file(path)

        data = self.transform(data)
        target = self.target_transform(target)
        return data, target


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, data_root, sub_meta, cl, transform=transforms.ToTensor(), target_transform=None, **kwargs):
        self.data_root = data_root
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        if target_transform is None:
            self.target_transform = transforms.Compose([Target(), ])
        else:
            self.target_transform = target_transform
        self.load_img = False
        if 'load_img' in kwargs.keys():
            self.load_img = kwargs['load_img']

    def __getitem__(self, i):
        # print( '%d -%d' %(self.cl,i))
        file_path = os.path.join(self.data_root, self.sub_meta[i])
        # print(file_path)

        # if self.load_img:
        #     data = Image.open(file_path).convert('RGB')
        # else:
        #     data = load_dict(file_path)

        data = load_file(file_path)
        data = self.transform(data)
        target = self.target_transform(self.cl)
        return data, target

    def __len__(self):
        return len(self.sub_meta)


class UserSetDataset:
    def __init__(self, data_file, data_root, users, n_way, n_shot, n_query, transform, mode, percentage, regression=False,
                 **kwargs):

        # calculate batch size for each class
        self.batch_size = n_shot + n_query
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_user = len(users)
        self.users = users
        self.percentage = percentage
        self.regression = regression
        self.set_mode(mode)
        # generate class list
        self.cl_list = [cl for cl in range(n_way)]
        if not self.regression:
            self.meta = load_dict(data_file)
        else:
            self.meta = load_dict(data_file)['data']
            self.meta_label = load_dict(data_file)['label']

        self.sub_meta = defaultdict(list)
        self.user_meta = defaultdict(defaultdict)
        self.user_meta_label = defaultdict(defaultdict)
        for u in users:
            for cl in self.cl_list:
                length = int(len(self.meta[u][cl]) * self.percentage)
                self.user_meta[cl][u] = self.meta[u][cl][:length]
                self.sub_meta[cl].extend(self.meta[u][cl][:length])
                if self.regression:
                    self.user_meta_label[cl][u] = self.meta_label[u][cl][:length]

        self.sub_dataloader = []
        self.samplers = []
        sub_data_loader_params = dict(batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)

        for cl in self.cl_list:
            if mode in ['p', 'd']:
                if not regression:
                    sampler = RandomUserSampler(user_meta=self.user_meta[cl], n_shot=self.n_shot,
                                                n_query=self.n_query, n_user=self.n_user)
                else:
                    sampler = RegressionSampler(user_meta=self.user_meta[cl], user_meta_label=self.user_meta_label[cl],
                                                n_shot=self.n_shot, n_query=self.n_query, n_user=self.n_user)
            else:
                sampler = RandomUserSamplerIndependent(user_meta=self.user_meta[cl], n_shot=self.n_shot,
                                                       n_query=self.n_query, n_user_support=self.n_user_support,
                                                       n_user_query=self.n_user_query)
            sub_dataset = SubDataset(data_root, self.sub_meta[cl], cl, transform=transform, **kwargs)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, sampler=sampler, **sub_data_loader_params))
            self.samplers.append(sampler)

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'p':
            self.n_user = 1
            if len(self.users) < self.n_user:
                raise ValueError("Number of users is not enough to generate data!")

        elif mode == 'd':
            self.n_user = 5
            if len(self.users) < self.n_user:
                raise ValueError("Number of users is not enough to generate data!")

        elif mode == 'i':
            self.n_user_support = 5
            self.n_user_query = 1
            if len(self.users) < 2:
                raise ValueError("Number of users is not enough to generate data!")

        else:
            raise ValueError('There is no mode {} in system'.format(mode))


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, seed, samplers, mode):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.seed = seed
        self.samplers = samplers
        self.mode = mode
        self.sampled_users = []

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # np.random.seed(self.seed[i])
            set_random_seed(self.seed[i])
            self.mode_sampler(self.mode, self.samplers)
            yield torch.randperm(self.n_classes)[:self.n_way]

    def mode_sampler(self, mode, samplers):
        sampler = samplers[0]
        if mode in ['p', 'd']:
            users_select = np.random.choice(sampler.users, size=sampler.n_user, replace=False)
            self.sampled_users.append(users_select)
            print('Support users: {}, Query users: {}'.format(users_select, users_select))
            for s in samplers:
                s.set_users(users_select)
        elif mode == 'i':
            if len(sampler.users) < sampler.n_user_query + sampler.n_user_support:
                users_select = np.random.choice(sampler.users, size=len(sampler.users), replace=False)
                support_users = users_select[:-1]
                while True:
                    size = min(sampler.n_user_query + sampler.n_user_support - len(users_select), len(support_users))
                    users_select = np.concatenate(
                        (np.random.choice(support_users, size=size, replace=False), users_select))
                    if len(users_select) == sampler.n_user_query + sampler.n_user_support:
                        break
            else:
                users_select = np.random.choice(sampler.users, size=(sampler.n_user_query + sampler.n_user_support),
                                                replace=False)
            self.sampled_users.append(users_select)
            users_support = users_select[:sampler.n_user_support]
            users_query = users_select[-sampler.n_user_query:]
            print('Support users: {}, Query users: {}'.format(users_support, users_query))
            for s in samplers:
                s.set_users(users_support, users_query)


class EpisodicBatchSamplerMeta(object):
    def __init__(self, n_classes, n_way, n_episodes, seed, samplers, mode, user_list):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.seed = seed
        self.samplers = samplers
        self.mode = mode
        user_list.sort(key=lambda x: int(x))
        self.maml_users = user_list
        self.n_user = len(user_list)
        if mode in ['meta', 'half-meta']:
            self.num_tasks = self.n_user * 2
        else:
            self.num_tasks = self.n_user

        self.id = 0
        self.maml_id = 0

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            if self.mode == 'meta':
                if self.id % self.num_tasks < self.n_user:
                    mode = 'maml'
                else:
                    mode = 'meta'
            elif self.mode == 'half-meta':
                if self.id % self.num_tasks < self.n_user / 2:
                    mode = 'maml'
                elif self.id % self.num_tasks >= self.n_user and self.id % self.num_tasks < self.n_user / 2 * 3:
                    mode = 'maml'
                else:
                    mode = 'meta'
            elif self.mode == 'quarter-meta':
                users = np.arange(self.n_user).astype(np.int)
                q1, q2, q3, q4 = np.array_split(users, 4)
                if self.id % self.num_tasks in q1 or self.id % self.num_tasks in q3:
                    mode = 'maml'
                else:
                    mode = 'meta'
            elif self.mode == 'maml':
                mode = 'maml'
            else:
                raise NotImplementedError("Not support for {} mode".format(self.mode))

            set_random_seed(self.seed[i])
            self.mode_sampler(mode, self.samplers)
            self.id += 1
            if mode == 'maml':
                self.maml_id += 1
            yield torch.randperm(self.n_classes)[:self.n_way]

    def mode_sampler(self, mode, samplers):
        if mode == 'meta':
            num_users = len(self.maml_users)
            if num_users < self.n_way:
                users_select = np.random.choice(self.maml_users, size=num_users, replace=False)
                while True:
                    size = min(self.n_way - len(users_select), num_users)
                    users_select = np.concatenate(
                        (users_select, np.random.choice(self.maml_users, size=size,
                                                        replace=False)))
                    if len(users_select) == self.n_way:
                        break
            else:
                users_select = np.random.choice(self.maml_users, size=self.n_way, replace=False)
            print('Meta >> Support users: {}, Query users: {}'.format(users_select, users_select))
            for i, s in enumerate(samplers):
                s.set_users(users_select[i])
        elif mode == 'maml':
            users_select = self.maml_users[self.maml_id % self.n_user]
            print('MAML >> Support users: {}, Query users: {}'.format(users_select, users_select))
            for s in samplers:
                s.set_users(users_select)


class RandomUserSampler(Sampler):
    def __init__(self, user_meta, n_shot, n_query, n_user, **kwargs):
        self.user_meta = user_meta
        self.n_user = n_user
        self.users = [u for u in self.user_meta.keys()]
        self.n_shot = n_shot
        self.n_query = n_query
        self.support_size = self.n_shot // self.n_user
        self.query_size = self.n_query // self.n_user
        self.users_select = []

    def set_users(self, users):
        self.users_select = users

    def __len__(self):
        return self.n_shot + self.n_query

    def __iter__(self):
        support_idxs = []
        query_idxs = []
        final_idxs = []
        u_startid = 0
        for u in self.user_meta.keys():
            if u in self.users_select:
                perm = torch.randperm(len(self.user_meta[u])) + u_startid
                sidx = perm[:self.support_size].tolist()
                qidx = perm[-self.query_size:].tolist()
                support_idxs.extend(sidx)
                query_idxs.extend(qidx)
            u_startid += len(self.user_meta[u])
        final_idxs.extend(support_idxs)
        final_idxs.extend(query_idxs)
        # print("Index: {}".format(final_idxs))
        return iter(final_idxs)


class RandomUserSamplerIndependent(Sampler):
    def __init__(self, user_meta, n_shot, n_query, n_user_support, n_user_query, **kwargs):
        self.user_meta = user_meta
        self.users = [u for u in self.user_meta.keys()]
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_user_support = n_user_support
        self.n_user_query = n_user_query
        self.support_size = self.n_shot // self.n_user_support
        self.query_size = self.n_query // self.n_user_query
        self.users_support = []
        self.users_query = []

    def set_users(self, us, uq):
        self.users_support = us
        self.users_query = uq

    def __len__(self):
        return self.n_shot + self.n_query

    def __iter__(self):
        support_idxs = []
        query_idxs = []
        final_idxs = []
        u_startid = 0

        for u in self.user_meta.keys():
            for _ in range(len(np.where(self.users_support == u)[0])):
                perm = torch.randperm(len(self.user_meta[u])) + u_startid
                sidx = perm[:self.support_size].tolist()
                support_idxs.extend(sidx)

            if u in self.users_query:
                perm = torch.randperm(len(self.user_meta[u])) + u_startid
                qidx = perm[:self.query_size].tolist()
                query_idxs.extend(qidx)
            u_startid += len(self.user_meta[u])

        final_idxs.extend(support_idxs)
        final_idxs.extend(query_idxs)
        # print("Index: {}".format(final_idxs))
        return iter(final_idxs)


class RegressionSampler(Sampler):
    def __init__(self, user_meta, user_meta_label, n_shot, n_query, n_user, **kwargs):
        self.user_meta = user_meta
        self.user_label_meta = user_meta_label
        self.users = [u for u in self.user_meta.keys()]
        self.n_shot = n_shot
        self.n_user = n_user
        self.n_query = n_query
        #sbp = 0
        #dbp = 1
        self.label_index = 1
        self.users_select = []

    def set_users(self, users):
        self.users_select = users

    def __len__(self):
        return self.n_shot + self.n_query

    def __iter__(self):
        support_idxs = []
        query_idxs = []
        final_idxs = []
        u_startid = 0
        for u in self.user_meta.keys():
            if u in self.users_select:
                # support
                labels = [x[self.label_index] for x in self.user_label_meta[u]]
                sorted_indices = np.argsort(labels)
                index_bins = np.array_split(sorted_indices, self.n_shot)
                for index in index_bins:
                    idx = np.random.choice(index, 1)[0] + u_startid
                    support_idxs.append(idx)
                    sorted_indices = sorted_indices[sorted_indices!=idx]

                # query
                index_bins = np.array_split(sorted_indices, self.n_query)
                for index in index_bins:
                    idx = np.random.choice(index, 1)[0] + u_startid
                    query_idxs.append(idx)
                labels = np.asarray(labels)
                ps_idx = np.asarray(support_idxs) - u_startid
                qs_idx = np.asarray(query_idxs) - u_startid
                print('Support labels: {}, Query labels:{}'.format(labels[ps_idx], labels[qs_idx]))

            u_startid += len(self.user_meta[u])
        final_idxs.extend(support_idxs)
        final_idxs.extend(query_idxs)
        # print("Index: {}".format(final_idxs))
        return iter(final_idxs)


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        path (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, path="", label=0, user=0, classname=""):
        assert isinstance(path, str)
        assert check_isfile(path)

        self._path = path
        self._label = label
        self._user = user
        self._classname = classname

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return self._label

    @property
    def user(self):
        return self._user

    @property
    def classname(self):
        return self._classname
