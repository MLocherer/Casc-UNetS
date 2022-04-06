r"""
    Copyright (C) 2022  Mark Locherer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
sys.path.append("pymodules")
from skimage.exposure import equalize_adapthist
import skimage
import skimage.transform
import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as TF


torch.manual_seed(17)

# https://pytorch.org/vision/0.8/transforms.html
# image augmentation w/ skimage https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/


def feature_scaling(input_array):
    r"""
    applies min-max feature scaling.
    Args:
        input_array: input array

    Returns: scaled array b/w [0, 1.0]

    """
    if torch.is_tensor(input_array):
        return (input_array - torch.min(input_array)) / (torch.max(input_array) - torch.min(input_array))

    return (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array))


class Compose(object):
    r"""
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    Using this Compose object implies, that always image and target are affected by a transfrom in the Compose object.
    Therefore, all transforms have to return both, namely, image and target as tuple
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    r"""
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __call__(self, image, target):
        # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C
        # x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr,
        # RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
        image = TF.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.uint8)
        return image, target


def pad_circular(x, pad):
    r"""
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x


class EquAdaptHist(object):
    r"""
    https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    """

    def __call__(self, image, target):
        if torch.is_tensor(image):
            arr = np.array(feature_scaling(image.numpy()) * 255, dtype=np.uint8).squeeze()
            return TF.to_tensor(equalize_adapthist(arr)), target
        else:
            arr = np.array(feature_scaling(image) * 255, dtype=np.uint8).squeeze()
            return equalize_adapthist(arr), target


class RandomHorizontalFlip(object):
    r"""
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            target = TF.vflip(target)
        return image, target


class Resize(object):
    r"""
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TF.resize(image, self.size)
        target = TF.resize(target, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, target


class RandomGaussianNoise(object):
    r"""
    This class adds random gaussian noise w/ mean = 0 and standard deviation std = std
    to an image w/ dimensions [..., H, W]
    """
    def __init__(self, noise_ammount):
        self.noise_ammount = noise_ammount

    def __call__(self, image, target):
        # Returns a tensor with the same size as input that is filled with random numbers from a normal distribution
        # with mean 0 and variance 1. torch.randn_like(input) is equivalent to torch.randn(input.size(),
        # dtype=input.dtype, layout=input.layout, device=input.device).
        return image + (torch.randn_like(image) * self.noise_ammount), target


class RandomRotation(object):
    r"""
    This class rotates a given image and target by a random number in the range of [-degrees, degrees]
    specified in degrees. It uses the function skimage.transform.rotate with mode 'wrap' that fills the
    points outside of the boundaries with the remaining pixels.
    (https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/)
    """
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        rotation = T.RandomRotation.get_params([-self.degrees, self.degrees])

        if rotation < 0:
            rotation += 360.0

        image_r = torch.zeros_like(image)
        for c in range(image.shape[0]):
            # rotate and wrap each image channel individually
            image_r[c] = TF.to_tensor(skimage.transform.rotate(image[c].numpy(), rotation, mode='wrap')).squeeze()

        target_r = TF.rotate(target, rotation)

        return image_r, target_r


class RandomCrop(object):
    r"""
    Source: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        # ReflectionPad2d: Pads the input tensor using the reflection of the input boundary.
        self.padder = nn.ReflectionPad2d(crop_size // 2)

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, (image.shape[1] - self.crop_size, image.shape[2] - self.crop_size))
        image = self.padder((TF.crop(image, *crop_params)).reshape(image.shape[0], 1, image.shape[1] - self.crop_size, image.shape[2] - self.crop_size))[:, 0, :, :]
        target = self.padder((TF.crop(target, *crop_params)).reshape(1, target.shape[0], target.shape[1] - self.crop_size, target.shape[2] - self.crop_size).double())[0].int()
        return image, target
