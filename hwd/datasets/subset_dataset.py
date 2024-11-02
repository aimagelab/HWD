from .base_dataset import BaseDataset
import numpy as np


class SubsetDataset:
    def __init__(self, dataset, max_samples):
        self.dataset = dataset
        self.max_samples = max_samples
        assert len(self.dataset) > 0 and self.max_samples <= len(self.dataset)
        self.indices = np.random.choice(len(self.dataset), self.max_samples, replace=False)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
