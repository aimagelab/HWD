from typing import Any
from PIL import Image
import numpy as np
import torch
import math
from torchvision.transforms import Compose, ToTensor, ColorJitter


class CropWhite:
    def __call__(self, img) -> Any:
        img_width, img_height = img.size
        tmp = np.array(img.convert('L')).mean(0)
        img_width = img_width - (tmp >= 250)[::-1].argmin()
        img_width = min(img_width, img_height)
        img = img.crop((0, 0, img_width, img_height))
        return img

class ResizeHeight:
    def __init__(self, height, interpolation=Image.NEAREST):
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        return img.resize((int(self.height * w / h), self.height), self.interpolation)


class PaddingMin:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        c, w, h = img.shape
        width = max(self.width, w)
        height = max(self.height, h)
        return torch.nn.functional.pad(img, (0, height - h, 0, width - w), mode='constant', value=0)


class CropStart:
    def __init__(self, width):
        self.width = width

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, self.width, h))


class CropStartSquare:
    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, h, h))


class ResizeSquare:
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class ToNumpy:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        elif isinstance(img, Image.Image):
            return np.array(img)
        elif isinstance(img, torch.Tensor):
            return img.numpy()
        else:
            raise TypeError(f'Unknown type: {type(img)}')


class Flatten:
    def __call__(self, img):
        return img.reshape(-1)
    

class ToInceptionV3Input:
    def __init__(self, size=299):
        self.size = size

    def __call__(self, x):
        h_rep = math.ceil(self.size / x.shape[1])
        w_rep = math.ceil(self.size / x.shape[2])
        return x.repeat(1, h_rep, w_rep)[:, :self.size, :self.size]


fid_ganwriting_transforms = Compose([
    CropStart(64),
    ResizeSquare(64),
    ToTensor()
])


def fid_ganwriting_color_transforms(val=0.5):
    return Compose([
        CropStart(64),
        ResizeSquare(64),
        ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
        ToTensor()
    ])


fid_our_transforms = Compose([
    CropStartSquare(),
    ResizeSquare(64),
    ToTensor()
])

fid_whole_transforms = Compose([
    ResizeHeight(299),
    ToTensor()
])


def fid_our_color_transforms(val=0.5):
    return Compose([
        CropStartSquare(),
        ResizeSquare(64),
        ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
        ToTensor()
    ])


gs_transforms = Compose([
    CropStart(64),
    # ResizeSquare(64),
    ToNumpy(),
    Flatten()
])

fred_transforms = Compose([
    ResizeHeight(32),
    ToTensor()
])

hwd_transforms = Compose([
    ResizeHeight(32),
    ToTensor(),
    PaddingMin(32, 32),
])

fved_beginning_transforms = Compose([
    ResizeHeight(32),
    CropStartSquare(),
    ToTensor(),
    PaddingMin(32, 32),
])

fred_64_transforms = Compose([
    ResizeHeight(64),
    ToTensor()
])


def fred_color_transforms(val=0.5):
    return Compose([
        ResizeHeight(32),
        ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
        ToTensor()
    ])
