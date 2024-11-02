from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class LeopardiDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, nameset)

        self.all_author_ids = [0, ]
        self.author_ids = self.all_author_ids

        self.imgs = [p for p in Path(path).rglob(f'*.jpg')]
        assert len(self.imgs) > 0, 'No images found.'
        self.labels = [0, ] * len(self.imgs)

