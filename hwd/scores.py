from .datasets.transforms import ResizeHeight, ToTensor, PaddingSquareHeight, Compose, CropStartSquare, ResizeSquare, OtsuBinarization
from .metrics.backbones import VGG16Backbone, InceptionV3Backbone, TrOCRBackbone, ActivationsVGG16Backbone, GeometryBackbone
from .metrics.distances import EuclideanDistance, FrechetDistance, MaximumMeanDiscrepancy, LPIPSDistance, IntraLPIPSDistance
from .metrics.gs import geom_score
from .metrics.base_score import BaseScore
from torchmetrics.text import CharErrorRate
from PIL import Image


VGG16_10400_URL = 'https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth'
INCEPTION_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class HWDScore(BaseScore):
    def __init__(self, height=32):
        backbone = VGG16Backbone(VGG16_10400_URL, num_classes=10400)
        distance = EuclideanDistance()
        transforms = Compose([
            PaddingSquareHeight(),
            ResizeHeight(height),
            ToTensor(),
        ])
        super().__init__(backbone, distance, transforms)


class FIDScore(BaseScore):
    def __init__(self, height=32, raise_on_error=False, batch_size=128):
        backbone = InceptionV3Backbone(INCEPTION_URL, batch_size=batch_size)
        distance = FrechetDistance(raise_on_error=raise_on_error)
        transforms = Compose([
            CropStartSquare(),
            ResizeSquare(height),
            ToTensor()
        ])
        super().__init__(backbone, distance, transforms)


class BFIDScore(BaseScore):
    def __init__(self, height=32, raise_on_error=False, batch_size=128):
        backbone = InceptionV3Backbone(INCEPTION_URL, batch_size=batch_size)
        distance = FrechetDistance(raise_on_error=raise_on_error)
        transforms = Compose([
            OtsuBinarization(),
            CropStartSquare(),
            ResizeSquare(height),
            ToTensor()
        ])
        super().__init__(backbone, distance, transforms)


class KIDScore(BaseScore):
    def __init__(self, height=32, batch_size=128, **kwargs):
        backbone = InceptionV3Backbone(INCEPTION_URL, batch_size=batch_size)
        distance = MaximumMeanDiscrepancy(**kwargs)
        transforms = Compose([
            CropStartSquare(),
            ResizeSquare(height),
            ToTensor()
        ])
        super().__init__(backbone, distance, transforms)


class BKIDScore(BaseScore):
    def __init__(self, height=32, batch_size=128, **kwargs):
        backbone = InceptionV3Backbone(INCEPTION_URL, batch_size=batch_size)
        distance = MaximumMeanDiscrepancy(**kwargs)
        transforms = Compose([
            OtsuBinarization(),
            CropStartSquare(),
            ResizeSquare(height),
            ToTensor()
        ])
        super().__init__(backbone, distance, transforms)


class CERScore(BaseScore):
    def __init__(self, height=64, **kwargs):
        backbone = TrOCRBackbone(**kwargs)
        distance = CharErrorRate()
        transforms = Compose([
            ResizeHeight(height),
        ])
        super().__init__(backbone, distance, transforms)
    
    def __call__(self, dataset, **kwargs) -> float:
        assert all(l is not None for l in dataset.labels), "This dataset does not contain any labels"
        preds, labels, authors = self.digest(dataset, **kwargs)
        return self.distance(preds, labels).item()
    

class LPIPSScore(BaseScore):
    def __init__(self, height=32):
        backbone = ActivationsVGG16Backbone(VGG16_10400_URL, num_classes=10400, batch_size=1)
        distance = LPIPSDistance()
        transforms = Compose([
            PaddingSquareHeight(),
            ResizeHeight(height),
            ToTensor(),
        ])
        super().__init__(backbone, distance, transforms)

    def _pad_to_same_size(self, img_1, img_2, padding_color=(255, 255, 255)):
        # Get the dimensions of both images
        width1, height1 = img_1.size
        width2, height2 = img_2.size
        
        # Determine the target size based on the largest width and height
        target_width = max(width1, width2)
        target_height = max(height1, height2)
        
        # Function to pad an image to the target size
        def pad_image(img, target_width, target_height):
            new_img = Image.new("RGB", (target_width, target_height), padding_color)
            new_img.paste(img, ((target_width - img.width) // 2, (target_height - img.height) // 2))
            return new_img

        # Pad both images
        img_1_padded = pad_image(img_1, target_width, target_height)
        img_2_padded = pad_image(img_2, target_width, target_height)
        return img_1_padded, img_2_padded
    
    def _sync_dataset(self, dataset1, dataset2):
        imgs_1, imgs_2 = [], []
        for img_1, img_2 in zip(dataset1.imgs, dataset2.imgs):
            if not isinstance(img_1, Image.Image):
                assert img_1.relative_to(dataset1.path) == img_2.relative_to(dataset2.path)
                img_1, img_2 = Image.open(img_1), Image.open(img_2)
            img_1, img_2 = self._pad_to_same_size(img_1, img_2)
            imgs_1.append(img_1)
            imgs_2.append(img_2)
        dataset1.imgs = imgs_1
        dataset2.imgs = imgs_2
        return dataset1, dataset2
    
    def __call__(self, dataset1, dataset2, stream=True, **kwargs) -> float:
        if not stream:
            Warning.warn('The LPIPS score uses a lot of memory, please use the stream=True argument')
        dataset1, dataset2 = self._sync_dataset(dataset1, dataset2)
        return super().__call__(dataset1, dataset2, stream=stream, **kwargs)
    

class IntraLPIPSScore(BaseScore):
    def __init__(self, height=32, batch_size=32):
        backbone = ActivationsVGG16Backbone(VGG16_10400_URL, num_classes=10400, batch_size=batch_size)
        distance = IntraLPIPSDistance()
        transforms = Compose([
            PaddingSquareHeight(),
            ResizeHeight(height),
            ToTensor(),
        ])
        super().__init__(backbone, distance, transforms)
    
    def __call__(self, dataset, stream=True, **kwargs) -> float:
        assert hasattr(dataset, 'imgs_ids'), f'To use the {self.__class__.__name__} you have to provide and Unfolded datasets "dataset.unfold()"'
        if stream:
            data_stream = self.digest_stream(dataset, **kwargs)
            return self.distance.from_streams(data_stream, dataset.imgs_ids)
        else:
            data = self.digest(dataset, **kwargs)
            return self.distance(data, dataset.imgs_ids)
        

class GeometryScore(BaseScore):
    def __init__(self, height=32, max_workers=8, L_0=64, gamma=None, i_max=100, n=1000):
        backbone = GeometryBackbone(max_workers=max_workers, L_0=L_0, gamma=gamma, i_max=i_max, n=n)
        distance = geom_score
        transforms = Compose([
            PaddingSquareHeight(),
            ResizeHeight(height),
            ToTensor(),
        ])
        super().__init__(backbone, distance, transforms)

    def __call__(self, dataset1, dataset2, stream=False, **kwargs):
        assert stream is False, 'The Geometry score does not support streaming'
        assert hasattr(dataset1, 'imgs_ids'), f'To use the {self.__class__.__name__} you have to provide and Unfolded datasets "dataset.unfold()"'
        assert hasattr(dataset2, 'imgs_ids'), f'To use the {self.__class__.__name__} you have to provide and Unfolded datasets "dataset.unfold()"'
        data1 = self.digest(dataset1, **kwargs)
        data2 = self.digest(dataset2, **kwargs)
        return self.distance(data1, data2).item()
