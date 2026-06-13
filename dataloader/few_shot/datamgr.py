# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import dataloader.few_shot.additional_transforms as add_transforms
from dataloader.few_shot.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, CategoriesSampler, DAPNDataset
from abc import abstractmethod


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            # transform_list = ['Scale', 'CenterCrop', 'ToTensor', 'Normalize']
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, dataroot, image_size, n_way, n_support, n_query, n_eposide=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.dataroot = dataroot

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(data_file, self.batch_size, transform, self.dataroot)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=False)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class DAPNDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
        super(DAPNDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = DAPNDataset(data_file, transform)
        fsl_train_sampler = CategoriesSampler(dataset.label, 100,
                                              self.n_way, self.batch_size)
        data_loader_params = dict(batch_sampler=fsl_train_sampler, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__ == "__main__":
    # 'p' n_shot=10 , n_query=5
    # 'd' n_shot=10 , n_query=15
    # 'i' n_shot=10 , n_query=5

    base_file = 'D:\GoodProject\Embedded_AI\MultiProj/CloserLookFewShot/filelists/miniImagenet/base.json'
    n_shot = 10
    n_query = 15
    image_size = 224
    n_eposide = 10
    train_n_way = 6
    n_shot = 5

    train_few_shot_params = dict(n_way=train_n_way, n_support=n_shot)
    # base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
    # base_loader = base_datamgr.get_data_loader(base_file, True)
    b_f = 'base.json'
    source_metafile = '../../dataloader/few_shot/filelists/miniImagenet/' + b_f

    base_datamgr = DAPNDataManager(image_size, n_query=n_query, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(source_metafile, True )

    print(base_loader.__len__())
    for x, y in base_loader:
        # print(y)
        print(x.shape, y.shape)
        # print(torch.transpose(y, 0,1))
        # print(torch.reshape(y, (25,6)))

        # break
