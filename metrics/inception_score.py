import torch
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities.data import dim_zero_cat

from .base_score import BaseScore
import numpy as np
from torch.utils.data import DataLoader


class InceptionScore:
    def __init__(self, feature='logits_unbiased', splits=10, normalize=True):
        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        self.normalize = normalize
        self.splits = splits

    def __call__(self, dataset, batch_size=1, verbose=False):
        data1 = self.digest(dataset)
        return self.distance(data1)

    def digest(self, dataset, batch_size=1, verbose=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        features = []
        for i, (images, _) in enumerate(loader):
            images = (images * 255).byte() if self.normalize else images
            features.append(self.inception(images))
        return features

    def distance(self, data1):
        features = dim_zero_cat(data1)
        # random permute the features
        idx = torch.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()
