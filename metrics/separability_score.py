from .base_score import BaseScore
import torch


class SeparabilityScore(BaseScore):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def digest(self, dataset, **kwargs):
        return dataset

    def distance(self, data1, data2, **kwargs):
        good = torch.Tensor(data1).to(self.device)
        bad = torch.Tensor(data2).to(self.device)

        upper_bound = good.max()
        lower_bound = bad.min()

        if upper_bound < lower_bound:
            upper_bound, lower_bound = lower_bound, upper_bound

        X = torch.linspace(lower_bound, upper_bound, 1000).to(self.device)

        X_good = X.repeat(good.size(0), 1).T - good
        X_bad = bad - X.repeat(bad.size(0), 1).T

        X_score = torch.cat([X_good, X_bad], dim=1).sum(dim=1)

        print()
        return None


