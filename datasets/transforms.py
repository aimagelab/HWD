from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor


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

fid_ganwriting_tranforms = Compose([
    CropStart(64),
    ResizeSquare(64),
    ToTensor()
])

fid_our_tranforms = Compose([
    CropStartSquare(),
    ResizeSquare(64),
    ToTensor()
])

gs_tranforms = Compose([
    CropStart(64),
    # ResizeSquare(64),
    ToNumpy(),
    Flatten()
])

fred_tranforms = Compose([
    ResizeHeight(32),
    ToTensor()
])