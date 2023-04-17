import torch
from torchmetrics.image.kid import poly_mmd
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities.data import dim_zero_cat

from .kid_score import KIDScore
from .fred_score import FReDScore, collate_fn
import numpy as np
from torch.utils.data import DataLoader
import warnings


class KReDScore(FReDScore):
    def __init__(self, checkpoint_path='metrics/fred/resnet_18_pretrained.pth', device='cuda', reduction='mean',
                 layers=4, subsets=100, subset_size=1000, degree=3, gamma=None, coef=1.0):
        super().__init__(checkpoint_path, device, reduction, layers)
        self.subsets = subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma
        self.coef = coef

    def digest(self, dataset, batch_size=1, verbose=False):
        if batch_size > 1:
            warnings.warn('WARNING: With batch_size > 1 you compute an approximation of the score. '
                          'Use batch_size=1 to compute the official score.')
            dataset.sort(verbose)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return self.get_activations(loader, verbose)

    def distance(self, data1, data2, **kwargs):
        features1 = dim_zero_cat(data1)
        features2 = dim_zero_cat(data2)

        n_samples_real = features1.shape[0]
        if n_samples_real < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")
        n_samples_fake = features2.shape[0]
        if n_samples_fake < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")

        kid_scores_ = []
        for _ in range(self.subsets):
            perm = torch.randperm(n_samples_real)
            f_real = features1[perm[: self.subset_size]]
            perm = torch.randperm(n_samples_fake)
            f_fake = features2[perm[: self.subset_size]]

            o = poly_mmd(f_real, f_fake, self.degree, self.gamma, self.coef)
            kid_scores_.append(o)
        kid_scores = torch.stack(kid_scores_)
        return kid_scores.mean(), kid_scores.std(unbiased=False)
