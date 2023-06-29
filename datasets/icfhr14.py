from .base_dataset import BaseDataset
from pathlib import Path


class ICFHR14Dataset(BaseDataset):
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

        self.path = Path(self.path) / 'lines'
        self.imgs = list(self.path.rglob('*.png'))

        self.author_ids = [0, ]
        self.labels = [0, ] * len(self.imgs)
