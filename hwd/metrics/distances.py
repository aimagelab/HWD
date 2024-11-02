import torch
from torch.linalg import inv
from torchmetrics.image.kid import poly_mmd
from torchmetrics.utilities.data import dim_zero_cat
from .base_score import BaseDistance, ProcessedDataset
from .fid.fid_score_crop64x64 import calculate_frechet_distance
from itertools import groupby, combinations


class EuclideanDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()
    
class FrechetDistance(BaseDistance):
    def __init__(self, raise_on_error=False):
        super().__init__()
        self.raise_on_error = raise_on_error

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        mu1 = torch.mean(data1.features.T, dim=1).cpu().numpy()
        sigma1 = torch.cov(data1.features.T).cpu().numpy()
        mu2 = torch.mean(data2.features.T, dim=1).cpu().numpy()
        sigma2 = torch.cov(data2.features.T).cpu().numpy()
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
class HammingDistance(BaseDistance):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        diff = torch.abs(tmp_1 - tmp_2)
        return torch.sum(diff > self.threshold).item()

class MahalanobisDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        mu = torch.mean(data1.features.T, dim=1)
        sigma = torch.cov(data1.features.T)
        sigma_inv = inv(sigma)
        cum_diff = torch.tensor(0.0)
        for i in range(data2.features.shape[0]):
            diff = data2.features[i].cpu().numpy() - mu
            cum_diff += torch.sqrt(diff.T @ sigma_inv @ diff)
        return cum_diff.item()

class CosineDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return 1 - torch.nn.functional.cosine_similarity(tmp_1, tmp_2, dim=1, eps=1e-8).item()

class MaximumMeanDiscrepancy(BaseDistance):
    def __init__(self, subsets=100, subset_size=1000, degree=3, gamma=None, coef=1.0):
        super().__init__()
        self.subsets = subsets
        self.subset_size = subset_size
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
    
    def __call__(self, data1, data2, **kwargs):
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
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
    

class LPIPSDistance(BaseDistance):
    def __init__(self):
        super().__init__()
        self.channels = [64, 128, 256, 512, 512]
        self.weights = [1e3 / n ** 2 for n in self.channels]
        self.chunk_sizes = [2048 * i for i in (32, 16, 8, 4, 1)]
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def count_consecutive_numbers(self, lst):
        return [len(list(group)) for _, group in groupby(lst)]
    
    def _style_loss(self, feat_1, feat_2):
        S = self._gram_matrix(feat_1)
        C = self._gram_matrix(feat_2)
        return self.mse_loss(S, C).mean((1, 2)).item()

    def _gram_matrix(self, F):
        b, c, hw = F.size()
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(hw)
        return G

    def __call__(self, data1, data2):
        assert len(data1.features) == len(data2.features)
        losses = []
        for feat_act_1, feat_act_2 in zip(data1.features, data2.features):
            loss_style = 0
            for feat_1, feat_2, weight in zip(feat_act_1, feat_act_2, self.weights):
                feat_1 = feat_1.flatten(2, 3)
                feat_2 = feat_2.flatten(2, 3)
                loss_style += self._style_loss(feat_1, feat_2) * weight
            losses.append(loss_style)
        return torch.tensor(losses).mean().item()
    

class IntraLPIPSDistance(LPIPSDistance):
    def count_consecutive_numbers(self, lst):
        return [len(list(group)) for _, group in groupby(lst)]
    
    def split_list(self, lst, split_sizes):
        result = []
        index = 0
        for size in split_sizes:
            result.append(lst[index:index + size])
            index += size
        return result

    def __call__(self, data, imgs_ids):
        assert len(data.features) == len(imgs_ids)
        groups = self.count_consecutive_numbers(imgs_ids)
        losses = []
        for features in self.split_list(data.features, groups):
            loss_style = 0
            features_T = list(zip(*features))
            for chunk, weight in zip(features_T, self.weights):
                for feat_1, feat_2 in combinations(chunk, 2):
                    feat_1 = feat_1.flatten(2, 3)
                    feat_2 = feat_2.flatten(2, 3)       
                    loss_style += self._style_loss(feat_1, feat_2) * weight
            losses.append(loss_style)
        return torch.tensor(losses).mean().item()