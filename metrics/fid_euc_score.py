from .fved_score import FVeDScore
from .fid_score import FIDScore
import torch


class FIDEucScore(FIDScore):
    def distance(self, data1, data2, **kwargs):
        tmp_1 = data1.features.mean(dim=0).unsqueeze(0)
        tmp_2 = data2.features.mean(dim=0).unsqueeze(0)
        return torch.cdist(tmp_1, tmp_2).item()
