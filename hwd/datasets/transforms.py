from typing import Any
from PIL import Image, ImageOps
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


class PaddingSquareHeight:
    def __call__(self, img):
        width, height = img.size
        
        # Check if padding is needed
        if width >= height:
            return img  # No padding needed if width is already greater than or equal to height
        
        # Calculate padding for the width
        new_width = height
        padding_left = (new_width - width) // 2
        padding_right = new_width - width - padding_left

        # Pad the image with white
        padded_image = ImageOps.expand(img, (padding_left, 0, padding_right, 0), fill="white")
        return padded_image


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


class GHTBinarization:
    def __init__(self):
        self.csum = lambda z: np.cumsum(z)[: -1]
        self.dsum = lambda z: np.cumsum(z[:: -1])[-2::-1]
        self.argmax = lambda x, f: np.mean(x[: -1][f == np.max(f)]) # Use the mean for ties.
        self.clip = lambda z: np.maximum(1e-30, z)

    def preliminaries(self, n, x):
        """ Some math that is shared across each algorithm ."""
        assert np.all(n >= 0)
        x = np.arange(len(n), dtype=n.dtype) if x is None else x
        assert np.all(x[1:] >= x[: -1])
        w0 = self.clip(self.csum(n))
        w1 = self.clip(self.dsum(n))
        p0 = w0 / (w0 + w1)
        p1 = w1 / (w0 + w1)
        mu0 = self.csum(n * x) / w0
        mu1 = self.dsum(n * x) / w1
        d0 = self.csum(n * x ** 2) - w0 * mu0 ** 2
        d1 = self.dsum(n * x ** 2) - w1 * mu1 ** 2
        return x, w0, w1, p0, p1, mu0, mu1, d0, d1

    def Otsu(self, n, x=None):
        """ Otsu ’s method ."""
        x, w0, w1, _, _, mu0, mu1, _, _ = self.preliminaries(n, x)
        o = w0 * w1 * (mu0 - mu1) ** 2
        return self.argmax(x, o), o

    def Otsu_equivalent(self, n, x=None):
        """ Equivalent to Otsu ’s method ."""
        x, _, _, _, _, _, _, d0, d1 = self.preliminaries(n, x)
        o = np.sum(n) * np.sum(n * x ** 2) - np.sum(n * x) ** 2 - np.sum(n) * (d0 + d1)
        return self.argmax(x, o), o

    def MET(self, n, x=None):
        """ Minimum Error Thresholding ."""
        x, w0, w1, _, _, _, _, d0, d1 = self.preliminaries(n, x)
        ell = (1 + w0 * np.log(self.clip(d0 / w0)) + w1 * np.log(self.clip(d1 / w1))
        - 2 * (w0 * np.log(self.clip(w0)) + w1 * np.log(self.clip(w1))))
        return self.argmax(x, - ell), ell # argmin()

    def wprctile(self, n, x=None, omega=0.5):
        """ Weighted percentile, with weighted median as default ."""
        assert omega >= 0 and omega <= 1
        x, _, _, p0, p1, _, _, _, _ = self.preliminaries(n, x)
        h = - omega * np.log(self.clip(p0)) - (1. - omega) * np.log(self.clip(p1))
        return self.argmax(x, -h), h # argmin()

    def GHT(self, n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
        """ Our generalization of the above algorithms ."""
        assert nu >= 0
        assert tau >= 0
        assert kappa >= 0
        assert omega >= 0 and omega <= 1
        x, w0, w1, p0, p1, _, _, d0, d1 = self.preliminaries(n, x)
        v0 = self.clip((p0 * nu * tau ** 2 + d0) /(p0 * nu + w0))
        v1 = self.clip((p1 * nu * tau ** 2 + d1) /(p1 * nu + w1))
        f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
        f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
        return self.argmax(x, f0 + f1), f0 + f1
    
    def im2hist(self, img, zero_extents=False):
        # Convert a PIL image to numpy grayscale, bin it, and optionally zero out the first and last bins.
        img = np.array(img)
        max_val = np.iinfo(img.dtype).max
        x = np.arange(max_val+1)
        e = np.arange(-0.5, max_val+1.5)
        assert len(img.shape) in [2, 3]
        img_bw = np.amax(img[...,:3], -1) if len(img.shape) == 3 else img
        n = np.histogram(img_bw, e)[0]
        if zero_extents:
            n[0] = 0
            n[-1] = 0
        return n, x, img_bw
    
    def __call__(self, img):
        n, x, img_bw = self.im2hist(img)
        level, _ = self.GHT(n, x)
        return Image.fromarray(img_bw > level).convert('RGB')
    

class OtsuBinarization(GHTBinarization):
    def __call__(self, img):
        n, x, img_bw = self.im2hist(img)
        level, _ = self.Otsu(n, x)
        return Image.fromarray(img_bw > level).convert('RGB')


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
