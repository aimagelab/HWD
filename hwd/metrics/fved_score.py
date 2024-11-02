from .base_score import ProcessedDataset
from .fred_score import FReDScore
import torch
from torchvision import models


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-2).squeeze(-1)

class AdjustDims(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.reshape(B, C)
        return x.reshape(B, C, -1)


class FVeDScore(FReDScore):
    def __init__(self, url='https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth', device='cpu', reduction='mean', layers=4):
        super().__init__(url, device, reduction, layers)

    def load_model(self):
        checkpoint = torch.hub.load_state_dict_from_url(self.url, progress=True, map_location=self.device)

        model = models.vgg16(num_classes=10400)
        model.load_state_dict(checkpoint)

        model_breakpoint = [4, 9, 16, 23, 30][self.layers] + 1
        modules = list(model.features.children())
        modules = modules[:model_breakpoint]

        if self.reduction == 'mean':
            modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))

        modules.append(AdjustDims())
        return torch.nn.Sequential(*modules)
