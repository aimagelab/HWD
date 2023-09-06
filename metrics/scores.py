from .base_score import BaseScore
from .distances import EuclideanDistance, FrechetDistance, MaximumMeanDiscrepancy
from .backbones import VGG16Backbone, InceptionV3Backbone
from datasets.transforms import ResizeHeight, ToTensor, PaddingMin, Compose, CropStartSquare, ResizeSquare

class HWDScore(BaseScore):
    def __init__(self, device='cpu'):
        backbone = VGG16Backbone('https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth', device=device)
        distance = EuclideanDistance()
        transforms = Compose([
            ResizeHeight(32),
            ToTensor(),
            PaddingMin(32, 32),
        ])
        super().__init__(backbone, distance, transforms)

class FIDScore(BaseScore):
    def __init__(self, raise_on_error=False, batch_size=128, device='cpu'):
        backbone = InceptionV3Backbone('https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth',
                                       batch_size=batch_size,
                                       device=device)
        distance = FrechetDistance(raise_on_error=raise_on_error)
        transforms = Compose([
            CropStartSquare(),
            ResizeSquare(64),
            ToTensor()
        ])
        super().__init__(backbone, distance, transforms)
        
class KIDScore(BaseScore):
    def __init__(self, batch_size=128, device='cpu', **kwargs):
        backbone = InceptionV3Backbone('https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth',
                                       batch_size=batch_size,
                                       device=device)
        distance = MaximumMeanDiscrepancy(**kwargs)
        transforms = Compose([
            CropStartSquare(),
            ResizeSquare(64),
            ToTensor()
        ])
        super().__init__(backbone, distance, transforms)