from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class RimesDataset(BaseDataset):
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
        super().__init__(path, transform, author_ids, nameset, max_samples)

        self.imgs = [p for p in Path(path).glob(f'lines/*')]

        self.all_author_ids = sorted(list(set(self.labels)))
        if author_ids is None:
            self.author_ids = self.all_author_ids

        self.imgs = [img for img in self.imgs if img.stem.split('-')[1] in self.author_ids]
        self.labels = [img.stem.split('-')[1] for img in self.imgs]

        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]
