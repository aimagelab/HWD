from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random

class KHATTDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        super().__init__(path, transform, author_ids, nameset, max_samples)

        path = Path(path)
        self.imgs = list(path.rglob('*.jpg'))
        self.all_author_ids = sorted(list(set([p.stem.split('_')[0] for p in self.imgs])))  # 838 instead of the 1000 in the paper
        if author_ids is None:
            self.author_ids = self.all_author_ids
        self.imgs = [p for p in self.imgs if p.stem.split('_')[0] in self.author_ids]
        self.labels = [img.stem.split('_')[0] for img in self.imgs]

        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]
