from .base_score import BaseScore
import numpy as np
import torch
import warnings
from torchvision import models
from pathlib import Path
from torch.utils.data import DataLoader
from .fred.frechet_distance import calculate_frechet_distance


def collate_fn(batch):
    imgs_width = [x[0].size(2) for x in batch]
    max_width = max(imgs_width)
    imgs = torch.stack([torch.nn.functional.pad(x[0], (0, max_width - x[0].size(2))) for x in batch], dim=0)
    ids = [x[1] for x in batch]
    return imgs, ids, torch.Tensor(imgs_width)


class UnFlatten(torch.nn.Module):
    def __init__(self, channels=512, height=1):
        super(UnFlatten, self).__init__()
        self.channels = channels
        self.height = height

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.channels, self.height, -1)


class FReDScore(BaseScore):
    def __init__(self, checkpoint_path='metrics/fred/resnet_18_pretrained.pth', device='cuda', reduction='mean',
                 layers=4):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.reduction = reduction

        checkpoint = torch.load(self.checkpoint_path)
        self.model = models.resnet18()
        self.model.fc = torch.nn.Identity()
        self.model.load_state_dict(checkpoint['model'], strict=False)

        if self.reduction == None:
            self.model.avgpool = torch.nn.Identity()
            self.model.fc = UnFlatten(2 ** (layers - 1) * 64, 1)
        elif self.reduction == 'mean':
            pass

        for i in range(4):
            if i >= layers:
                self.model.__dict__[f'layer{i + 1}'] = torch.nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_statistics_of_dataset(self, loader, verbose=False):
        act = self.get_activations(loader, verbose)
        mu = torch.mean(act, dim=1)
        sigma = torch.cov(act)

        return mu, sigma

    @torch.no_grad()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        pred_arr = []
        for i, (images, _, widths) in enumerate(loader):
            images = images.to(self.device)

            pred = self.model(images)

            if self.reduction == None:
                widths = widths.to(self.device)
                pred_widths = (widths / 32).ceil().long()
                pred = [t[:, :, :w] for t, w in zip(pred, pred_widths)]
                pred_arr.append(torch.cat(pred, dim=-1).flatten(start_dim=1).cpu())
            elif self.reduction == 'mean':
                pred_arr.append(pred.permute(1, 0).cpu())

            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)
        if verbose:
            print(' OK')

        return torch.cat(pred_arr, dim=1)

    def digest(self, dataset, batch_size=1, verbose=False):
        if batch_size > 1:
            warnings.warn('WARNING: With batch_size > 1 you compute an approximation of the score. '
                          'Use batch_size=1 to compute the official score.')
            dataset.sort(verbose)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return self.compute_statistics_of_dataset(loader, verbose)

    def distance(self, data1, data2, **kwargs):
        return calculate_frechet_distance(*data1, *data2)
