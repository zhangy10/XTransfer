from pyts.image import GramianAngularField
import scipy.signal as scisig
import numpy as np
from torch.nn.functional import interpolate
import torch
from PIL import Image
import cv2
import torch
from PIL import ImageEnhance


class STFT(object):
    """
    STFT transformer
    """

    def __init__(self, nperseg):
        self.nperseg = nperseg

    def __call__(self, data):
        X_stft = []
        num_channel = data.shape[0]
        for i in range(num_channel):
            channel_data = data[i]
            _, _, cx = scisig.stft(channel_data, nperseg=self.nperseg, noverlap=0, padded=True)
            X_stft.append(np.abs(cx))
        X_stft = np.stack(X_stft)
        return X_stft


class ToTensor(object):
    def __init__(self):
        return

    def __call__(self, data):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        return data


class GADF(object):
    """
    GramianAngularField transformer
    """

    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, data):
        img_size = min(data.shape[1], self.img_size)
        gadf_obj = GramianAngularField(image_size=img_size, method='difference')
        gadf_data = gadf_obj.fit_transform(data)
        gadf_data = torch.from_numpy(gadf_data).float()

        return gadf_data


class Resize(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, data):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        data = data[None, :]
        interp_data = interpolate(data, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        interp_data = torch.squeeze(interp_data, 0)
        return interp_data


class Resize_1D(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, data):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        data = data[None, :]
        # interp_data = interpolate(data, size=(self.img_size, data.size(3)), mode='bilinear', align_corners=False)
        interp_data = interpolate(data, size=self.img_size, mode='linear', align_corners=False)
        interp_data = torch.squeeze(interp_data, 0)
        return interp_data


class Raw(object):
    def __init__(self):
        return

    def __call__(self, data):
        data = torch.from_numpy(data).float()
        return data


def find_2d_shape(size):
    factors = []
    for i in range(1, int(np.sqrt(size)) + 1):
        if size % i == 0:
            factors.append(i)
            factors.append(size // i)

    # create the almost squared shape
    n = factors[-2]
    m = factors[-1]

    # swap the dimensions if necessary
    if m < n:
        n, m = m, n

    # return the resulting shape
    return (n, m)


class Raw2D(object):
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, data):
        # data = ((data - data.min()) * (1 / (data.max() - data.min()) * 1)).astype('uint8')
        # data = np.reshape(data, (10, 256))
        data = torch.from_numpy(data).float()
        data = torch.unsqueeze(data, self.axis)
        return data


class Raw2D_square(object):
    def __init__(self, axis=1):
        self.axis = axis
        self.h = None
        self.w = None

    def __call__(self, data):
        # data = ((data - data.min()) * (1 / (data.max() - data.min()) * 1)).astype('uint8')
        size = np.reshape(data, (-1)).size
        if self.h is None or self.h * self.w != size:
            self.h, self.w = find_2d_shape(size)
        data = np.reshape(data, (self.h, self.w))
        data = torch.from_numpy(data).float()
        data = torch.unsqueeze(data, self.axis)
        return data


class Raw2Image(object):
    def __init__(self):
        return

    def __call__(self, data):
        data = ((data - data.min()) * (1 / (data.max() - data.min()) * 255)).astype('uint8')
        # data = np.expand_dims(data, axis=0)
        img = cv2.applyColorMap(data, cv2.COLORMAP_JET)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.transpose(img, (2,0,1))
        img = Image.fromarray(img)
        return img


class Raw1D(object):
    def __init__(self):
        return

    def __call__(self, data):
        data = torch.from_numpy(data).float()
        return data


class Normalize1D(object):
    def __init__(self, raw_mean, raw_std):
        self.raw_mean = raw_mean
        self.raw_std = raw_std
        return

    def __call__(self, data):
        data = (data - self.raw_mean[0]) / self.raw_std[0]
        return data


class Target(object):
    def __init__(self):
        return

    def __call__(self, data):
        return data


transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
