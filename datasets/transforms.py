from PIL import Image
from torchvision.transforms import Compose, ToTensor


class ResizeHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        w, h = img.size
        return img.resize((int(self.height * w / h), self.height), Image.INTER_AREA)


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


ganwriting_fid_tranforms = Compose([
    CropStart(64),
    ResizeSquare(64),
    ToTensor()
])

our_fid_tranforms = Compose([
    CropStartSquare(),
    ResizeSquare(64),
    ToTensor()
])