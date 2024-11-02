import torch
from torchmetrics.image.kid import poly_mmd
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities.data import dim_zero_cat

from .base_score import BaseScore, ProcessedDataset
from torch.utils.data import DataLoader


class KIDScore(BaseScore):
    def __init__(self, feature=2048, subsets=100, subset_size=1000, degree=3, gamma=None, coef=1.0,
                 reset_real_features=True, normalize=True, device='cpu'):

        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        self.inception = self.inception.to(device)
        self.subsets = subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
        self.reset_real_features = reset_real_features
        self.normalize = normalize
        self.device = device

    def digest(self, dataset, batch_size=128, verbose=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ids, labels, features = self.get_activations(loader, verbose)
        return ProcessedDataset(ids, labels, features)

    @torch.no_grad()
    def get_activations(self, loader, verbose=False):
        self.inception.eval()

        features = []
        labels = []
        for i, (images, authors) in enumerate(loader):
            images = images.to(self.device)
            images = (images * 255).byte() if self.normalize else images
            features.append(self.inception(images))
            labels.append(list(authors))
            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)} ', end='', flush=True)
                print('OK') if i + 1 == len(loader) else None

        labels = sum(labels, [])
        features = torch.cat(features, dim=0)
        ids = torch.arange(len(labels), dtype=torch.long)
        return ids, labels, features

    def distance(self, data1, data2, **kwargs):
        features1 = dim_zero_cat(data1.features)
        features2 = dim_zero_cat(data2.features)

        n_samples_real = features1.shape[0]
        # if n_samples_real < self.subset_size:
        #     raise ValueError("Argument `subset_size` should be smaller than the number of samples")
        n_samples_fake = features2.shape[0]
        # if n_samples_fake < self.subset_size:
        #     raise ValueError("Argument `subset_size` should be smaller than the number of samples")
        subset_size = min(self.subset_size, n_samples_real, n_samples_fake)

        kid_scores_ = []
        for _ in range(self.subsets):
            perm = torch.randperm(n_samples_real)
            f_real = features1[perm[: subset_size]]
            perm = torch.randperm(n_samples_fake)
            f_fake = features2[perm[: subset_size]]

            o = poly_mmd(f_real, f_fake, self.degree, self.gamma, self.coef)
            kid_scores_.append(o)
        kid_scores = torch.stack(kid_scores_)
        return kid_scores.mean().item()
