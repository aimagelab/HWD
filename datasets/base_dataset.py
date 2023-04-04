import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        if nameset is not None:
            raise NotImplementedError('Nameset is not implemented yet.')

        self.path = path
        self.imgs = []
        self.labels = {}
        self.transform = transform
        self.author_ids = author_ids
        self.nameset = nameset
        self.max_samples = max_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class.
        """
        pass

    def __len__(self):
        return len(self.imgs)