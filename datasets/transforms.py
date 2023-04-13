from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, ColorJitter


class ResizeHeight:
    def __init__(self, height, interpolation=Image.NEAREST):
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        return img.resize((int(self.height * w / h), self.height), self.interpolation)


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


def fred_color_transforms(val=0.5):
    return Compose([
        ResizeHeight(32),
        ColorJitter(brightness=val, contrast=val, saturation=val, hue=val),
        ToTensor()
    ])
