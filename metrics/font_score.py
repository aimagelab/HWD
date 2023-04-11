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


class FontScore(BaseScore):
    def __init__(self, checkpoint_path='metrics/fred/resnet_18_pretrained.pth', device='cuda'):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

        checkpoint = torch.load(self.checkpoint_path)
        self.model = models.resnet18()
        # self.model.avgpool = torch.nn.Identity()
        self.model.fc = torch.nn.Identity()
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # self.model.fc = UnFlatten(512, 1)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    @torch.no_grad()
    def get_activations(loader, model, device, verbose=False):
        model.eval()

        pred_arr = []
        for i, (images, _, _) in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred_arr.append(pred.permute(1, 0).cpu())
            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)

        if verbose:
            print(' OK')

        return torch.cat(pred_arr, dim=1).mean(dim=1)

    def digest(self, dataset, batch_size=1, verbose=False):
        if batch_size > 1:
            warnings.warn('WARNING: With batch_size > 1 you compute an approximation of the score. '
                          'Use batch_size=1 to compute the official score.')
            dataset.sort(verbose)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return FontScore.get_activations(loader, self.model, self.device, verbose)

    def distance(self, data1, data2, **kwargs):
        return torch.cdist(data1.view(1, 1, -1), data2.view(1, 1, -1)).squeeze().item()

    def distance_batch(self, data, **kwargs):
        data = torch.stack(data, dim=0).unsqueeze(0)
        dist = torch.cdist(data, data).squeeze(0)
        idx = torch.triu_indices(dist.size(0), dist.size(1), 1)
        return dist[idx.unbind()].tolist()
