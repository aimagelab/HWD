import torch
from torchmetrics.image.kid import poly_mmd
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities.data import dim_zero_cat

from .base_score import BaseScore
import numpy as np
from torch.utils.data import DataLoader


class KIDScore(BaseScore):
    def __init__(self, feature=2048, subsets=100, subset_size=1000, degree=3, gamma=None, coef=1.0,
                 reset_real_features=True, normalize=True):

        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        self.subsets = subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
        self.reset_real_features = reset_real_features
        self.normalize = normalize

    def digest(self, dataset, batch_size=1, verbose=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features = []
        for i, (images, _) in enumerate(loader):
            images = (images * 255).byte() if self.normalize else images
            features.append(self.inception(images))
        return features

    def distance(self, data1, data2):
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


