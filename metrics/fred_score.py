from .base_score import BaseScore, ProcessedDataset
import numpy as np
import torch
import warnings
from torchvision import models
from pathlib import Path
from torch.utils.data import DataLoader
from .fred.frechet_distance import calculate_frechet_distance
from .fred.pyramidpooling import TemporalPyramidPooling


class UnFlatten(torch.nn.Module):
    def __init__(self, channels=512, height=1):
        super(UnFlatten, self).__init__()
        self.channels = channels
        self.height = height

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.channels, self.height, -1)


class FReDScore(BaseScore):
    def __init__(self, url='https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth', device='cpu', reduction='mean',
                 layers=4):
        super().__init__()
        self.url = url
        self.device = device
        self.reduction = reduction
        self.layers = layers
        self.model = self.load_model()
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path)

        model = models.resnet18()
        model.fc = torch.nn.Identity()
        model.load_state_dict(checkpoint['model'], strict=False)

        if self.reduction == None:
            model.avgpool = torch.nn.Identity()
            model.fc = UnFlatten(2 ** (self.layers - 1) * 64, 1)
        elif self.reduction == 'mean':
            pass
        elif self.reduction == 'tpp':
            model.avgpool = TemporalPyramidPooling((1, 2, 4))
            model.fc = torch.nn.Identity()

        for i in range(4):
            if i >= self.layers:
                model.__dict__[f'layer{i + 1}'] = torch.nn.Identity()
        return model

    @torch.no_grad()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        features = []
        labels = []
        ids = []
        for i, (images, authors, widths) in enumerate(loader):
            images = images.to(self.device)

            pred = self.model(images)
            pred = pred.squeeze(-2)

            if self.reduction == None:
                if pred.ndim < 3:
                    pred = pred.reshape(loader.batch_size, -1, 1)
                pred_width = pred.size(-1)
                labels.append(sum([[author] * pred_width for author in authors], []))
                ids.append(sum([[i * loader.batch_size + d] * pred_width for d in range(len(authors))], []))
                features.append(pred.permute(1, 0, 2).flatten(start_dim=1).cpu().T)
            elif self.reduction in ('mean', 'tpp'):
                pred = pred.unsqueeze(0) if pred.ndim == 1 else pred
                labels.append(authors)
                ids.append([i * loader.batch_size + d for d in range(len(authors))])
                features.append(pred.cpu())
            else:
                raise NotImplementedError

            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)
        if verbose:
            print(' OK')

        ids = torch.Tensor(sum(ids, [])).long()
        labels = sum(labels, [])
        features = torch.cat(features, dim=0)
        return ids, labels, features

    def digest(self, dataset, batch_size=1, verbose=False):
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
        mu1 = torch.mean(data1.features.T, dim=1).cpu().numpy()
        sigma1 = torch.cov(data1.features.T).cpu().numpy()
        mu2 = torch.mean(data2.features.T, dim=1).cpu().numpy()
        sigma2 = torch.cov(data2.features.T).cpu().numpy()
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
