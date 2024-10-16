from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class BaseDataset(Dataset):
    def __init__(self, path, transform=None, nameset=None, preprocess=None):
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
        self.labels = []
        self.transform = transform
        self.preprocess = preprocess
        self.nameset = nameset
        self.is_sorted = False
        self.author_ids = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class.
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB') if isinstance(img, Path) else img 
        label = self.labels[index]
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def sort(self, verbose=False):
        if self.is_sorted:
            return
        imgs_width = []
        for i, (img, label) in enumerate(self):
            imgs_width.append(img.size(2))
            if verbose:
                print(f'\rSorting {i + 1}/{len(self)} ', end='', flush=True)
        self.imgs = [x for _, x in sorted(zip(imgs_width, self.imgs), key=lambda pair: pair[0])]
        if verbose:
            print(' OK')
        self.is_sorted = True
