import numpy as np
import torch
import warnings
from torchvision import models
from pathlib import Path
from torch.utils.data import DataLoader
from .frechet_distance import calculate_frechet_distance

def compute_statistics_of_dataset(loader, model, device, verbose=False):
    act = get_activations(loader, model, device, verbose)
    mu = torch.mean(act, dim=1)
    sigma = torch.cov(act)

    return mu, sigma

@torch.no_grad()
def get_activations(loader, model, device, verbose=False):
    model.eval()

    pred_arr = []
    for i, (images, _, widths) in enumerate(loader):
        images = images.to(device)
        widths = widths.to(device)
        pred_widths = (widths / 32).ceil().long()

        pred = model(images)
        pred = [t[:, :, :w] for t, w in zip(pred, pred_widths)]

        pred_arr.append(torch.cat(pred, dim=-1).flatten(start_dim=1).cpu())

        if verbose:
            print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)

    if verbose:
        print(' OK')

    return torch.cat(pred_arr, dim=1)

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


class FReD:
    def __init__(self, dataset1, dataset2, checkpoint_path='metrics/fred/resnet_18_pretrained.pth', device='cuda'):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.checkpoint_path = checkpoint_path
        self.device = device

        checkpoint = torch.load(self.checkpoint_path)
        self.model = models.resnet18()
        self.model.avgpool = torch.nn.Identity()
        self.model.fc = torch.nn.Identity()
        self.model.load_state_dict(checkpoint['model'], strict=False)

        self.model.fc = UnFlatten(512, 1)
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute(self, batch_size=1, verbose=False):
        if batch_size > 1:
            warnings.warn('WARNING: With batch_size > 1 you compute an approximation of the score. '
                          'Use batch_size=1 to compute the official score.')
            self.dataset1.sort(verbose)
            self.dataset2.sort(verbose)

        loader1 = DataLoader(self.dataset1, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        loader2 = DataLoader(self.dataset2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        m1, s1 = compute_statistics_of_dataset(loader1, self.model, self.device, verbose)
        m2, s2 = compute_statistics_of_dataset(loader2, self.model, self.device, verbose)
        fred_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fred_value
