from .base_score import ProcessedDataset
from .fred_score import FReDScore
import torch
from torchvision import models
from torchvision.models.vgg import VGG16_Weights


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-2).squeeze(-1)

class AdjustDims(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.reshape(B, C)
        return x.reshape(B, C, -1)


class VontImageNetScore(FReDScore):
    def __init__(self, checkpoint_path='metrics/fved/vgg_16_pretrained.pth', device='cuda', reduction='mean', layers=4):
        super().__init__(checkpoint_path, device, reduction, layers)

    def distance(self, data1, data2, **kwargs):
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()

    def load_model(self):
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        model_breakpoint = [4, 9, 16, 23, 30][self.layers] + 1
        modules = list(model.features.children())
        modules = modules[:model_breakpoint]

        if self.reduction == 'mean':
            modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))

        modules.append(AdjustDims())
        return torch.nn.Sequential(*modules)
