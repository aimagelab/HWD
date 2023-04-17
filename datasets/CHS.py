from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class CHSDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        super().__init__(path, transform, author_ids, nameset, max_samples)

        path = Path(path)
        self.all_author_ids = sorted([p.stem for p in (path / 'data' / 'data' / 'test').iterdir()])
        if author_ids is None:
            self.author_ids = self.all_author_ids

        self.imgs = [p for author_id in self.author_ids for p in (path / 'data' / 'data').rglob(f'{author_id}/*.jpg')]
        self.labels = [p.parent for p in self.imgs]

        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]
