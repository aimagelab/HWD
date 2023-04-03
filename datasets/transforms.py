from PIL import Image


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
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size), Image.BILINEAR)
