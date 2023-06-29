from __future__ import annotations
import pickle
import torch


class ProcessedDataset:
    def __init__(self, ids, labels, features):
        self.ids = ids
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, labels) -> ProcessedDataset:
        if not isinstance(labels, (list, tuple, set)):
            labels = (labels,)
        mask = torch.Tensor([label in labels for label in self.labels])
        mask = mask.to(torch.bool)
        ids = self.ids[mask]
        labels = [label for label, m in zip(self.labels, mask) if m]
        features = self.features[mask]
        return ProcessedDataset(ids, labels, features)

    def subset(self, n) -> ProcessedDataset:
        unique_ids = torch.unique(self.ids)
        assert n <= len(unique_ids)
        mask = torch.randperm(len(unique_ids))[:n]
        mask = set(mask.tolist())
        mask = torch.Tensor([id.item() in mask for id in self.ids]).to(torch.bool)
        ids = self.ids[mask]
        labels = [label for label, m in zip(self.labels, mask) if m]
        features = self.features[mask]
        return ProcessedDataset(ids, labels, features)

    def split(self, ratio=0.5) -> tuple[ProcessedDataset, ProcessedDataset]:
        ids = torch.unique(self.ids)
        ids = ids[torch.randperm(len(ids))]
        split = int(len(ids) * ratio)
        ids1 = set(ids[:split].tolist())
        ids2 = set(ids[split:].tolist())
        mask1 = torch.Tensor([id.item() in ids1 for id in self.ids]).to(torch.bool)
        mask2 = torch.Tensor([id.item() in ids2 for id in self.ids]).to(torch.bool)
        dataset1 = ProcessedDataset(self.ids[mask1], [label for label, m in zip(self.labels, mask1) if m], self.features[mask1])
        dataset2 = ProcessedDataset(self.ids[mask2], [label for label, m in zip(self.labels, mask2) if m], self.features[mask2])
        assert len(dataset1) + len(dataset2) == len(self), f'{len(dataset1)} + {len(dataset2)} != {len(self)}'
        return dataset1, dataset2

    def save(self, path):
        data = {
            'ids': self.ids.cpu().numpy(),
            'labels': self.labels,
            'features': self.features.cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path) -> ProcessedDataset:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        data['ids'] = torch.Tensor(data['ids'])
        data['features'] = torch.Tensor(data['features'])
        return ProcessedDataset(data['ids'], data['labels'], data['features'])

    @property
    def device(self):
        assert self.features.device == self.ids.device
        return self.features.device

    def to(self, device):
        self.ids = self.ids.to(device)
        self.features = self.features.to(device)
        return self


class BaseScore:
    def __call__(self, dataset1, dataset2, **kwargs) -> float:
        """
        Compute the distance between two datasets
        :param dataset1: first dataset
        :param dataset2: second dataset
        :param kwargs: extra parameters to pass to the digest function
        :return: the distance between the two datasets
        """
        data1 = self.digest(dataset1, **kwargs)
        data2 = self.digest(dataset2, **kwargs)
        return self.distance(data1, data2)

    def digest(self, dataset, **kwargs) -> ProcessedDataset:
        raise NotImplementedError

    def distance(self, data1, data2) -> float:
        raise NotImplementedError
