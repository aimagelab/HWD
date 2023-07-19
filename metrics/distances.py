import torch
from torch.linalg import inv
from .base_score import BaseDistance, ProcessedDataset
from .fid.fid_score_crop64x64 import calculate_frechet_distance

class EuclideanDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs) -> float:
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()
    
class FrechetDistance(BaseDistance):
    def __init__(self, raise_on_error=False):
        super().__init__()
        self.raise_on_error = raise_on_error

    def __call__(self, data1, data2, **kwargs) -> float:
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

    def __call__(self, data1, data2, **kwargs) -> float:
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        diff = torch.abs(tmp_1 - tmp_2)
        return torch.sum(diff > self.threshold).item()

class MahalanobisDistance(BaseDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, data1, data2, **kwargs) -> float:
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

    def __call__(self, data1, data2, **kwargs) -> float:
        assert isinstance(data1, ProcessedDataset)
        assert isinstance(data2, ProcessedDataset)
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return 1 - torch.nn.functional.cosine_similarity(tmp_1, tmp_2, dim=1, eps=1e-8).item()