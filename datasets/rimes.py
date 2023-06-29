from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class RimesDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        """
               Args:
                   path (string): Path folder of the dataset.
                   transform (callable, optional): Optional transform to be applied
                       on a sample.
                   author_ids (list, optional): List of authors to consider.
                   nameset (string, optional): Name of the dataset.
                   max_samples (int, optional): Maximum number of samples to consider.
               """
        super().__init__(path, transform, nameset)

        self.path = Path(path) / 'lines'
        self.imgs = list(self.path.rglob('*.png'))

        self.author_ids = {img.stem.split('-')[1] for img in self.imgs}

        self.imgs = [img for img in self.imgs if img.stem.split('-')[1] in self.author_ids]
        self.author_ids = sorted(self.author_ids)
        self.labels = [img.stem.split('-')[1] for img in self.imgs]
