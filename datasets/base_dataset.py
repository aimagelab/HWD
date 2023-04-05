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
        self.is_sorted = False

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

    def sort(self, verbose=False):
        if self.is_sorted:
            return
        imgs_width = []
        for i, (img, label) in enumerate(self):
            imgs_width.append(img.size(2))
            if verbose:
                print(f'\rSorting {i + 1}/{len(self)}', end='', flush=True)
        self.imgs = [x for _, x in sorted(zip(imgs_width, self.imgs), key=lambda pair: pair[0])]
        if verbose:
            print(' OK')
        self.is_sorted = True
