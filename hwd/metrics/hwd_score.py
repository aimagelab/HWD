import torch
import warnings
from torchvision import models
from torch.utils.data import DataLoader
from datasets.transforms import hwd_transforms
from .base_score import BaseScore, ProcessedDataset


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-2).squeeze(-1)

class AdjustDims(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.reshape(B, C)
        return x.reshape(B, C, -1)

class UnFlatten(torch.nn.Module):
    def __init__(self, channels=512, height=1):
        super(UnFlatten, self).__init__()
        self.channels = channels
        self.height = height

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.channels, self.height, -1)


class HWDScore(BaseScore):
    def __init__(self, url='https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth', device='cpu'):
        super().__init__()
        self.url = url
        self.device = torch.device(device)
        self.model = self.load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transforms = hwd_transforms

    def load_model(self):
        model = models.vgg16(num_classes=10400)

        checkpoint = torch.hub.load_state_dict_from_url(self.url, progress=True, map_location=self.device)
        model.load_state_dict(checkpoint)

        modules = list(model.features.children())
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(AdjustDims())
        return torch.nn.Sequential(*modules)

    @torch.no_grad()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        features, labels, ids = [], [], []
        for i, (images, authors, _) in enumerate(loader):
            images = images.to(self.device)

            pred = self.model(images)
            pred = pred.squeeze(-2)

            pred = pred.unsqueeze(0) if pred.ndim == 1 else pred
            labels.append(authors)
            ids.append([i * loader.batch_size + d for d in range(len(authors))])
            features.append(pred.cpu())

            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)
        if verbose:
            print(' OK')

        ids = torch.Tensor(sum(ids, [])).long()
        labels = sum(labels, [])
        features = torch.cat(features, dim=0)
        return ids, labels, features

    def digest(self, dataset, batch_size=1, verbose=False):
        dataset.transform = self.transforms
        if batch_size > 1:
            warnings.warn('WARNING: With batch_size > 1 you compute an approximation of the score. '
                          'Use batch_size=1 to compute the official score.')
            print('Sorting dataset to mitigate the "batch_size > 0" effect')
            dataset.sort(verbose)
            print('Sorted dataset')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        ids, labels, features = self.get_activations(loader, verbose)
        return ProcessedDataset(ids, labels, features)

    def collate_fn(self, batch):
        imgs_width = [x[0].size(2) for x in batch]
        max_width = max(imgs_width)
        imgs = torch.stack([torch.nn.functional.pad(x[0], (0, max_width - x[0].size(2))) for x in batch], dim=0)
        ids = [x[1] for x in batch]
        return imgs, ids, torch.Tensor(imgs_width)

    def distance(self, data1, data2, **kwargs):
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()
