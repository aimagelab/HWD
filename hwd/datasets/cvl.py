from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class CVLDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        raise NotImplementedError
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

        # self.labels = {str(author_id): int(label) for label, author_id in enumerate(self.all_author_ids)}
        self.author_ids = sorted([p.stem for p in Path(path).glob('*/lines/*')])

        self.imgs = [p for p in Path(path).rglob(f'lines/*/*')]
        self.labels = [img.parent.stem for img in self.imgs]
