import torch
from torch.utils.data import Dataset

class MetricDataset(Dataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
        """
        if nameset is not None:
            raise NotImplementedError('Nameset is not implemented yet.')
        
        self.imgs = []
        self.labels = []

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