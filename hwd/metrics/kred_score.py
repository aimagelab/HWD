import torch
from torchmetrics.image.kid import poly_mmd
from torchmetrics.utilities.data import dim_zero_cat
from .fred_score import FReDScore


class KReDScore(FReDScore):
    def __init__(self, checkpoint_path='metrics/fred/resnet_18_pretrained.pth', device='cuda', reduction='mean',
                 layers=4, subsets=100, subset_size=1000, degree=3, gamma=None, coef=1.0):
        super().__init__(checkpoint_path, device, reduction, layers)
        self.subsets = subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma
        self.coef = coef

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
