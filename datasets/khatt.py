from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class KHATTDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        super().__init__(path, transform, nameset)

        path = Path(path)
        self.imgs = list(path.rglob('*.jpg'))
        self.author_ids = sorted(
            list(set([p.stem.split('_')[0] for p in self.imgs])))  # 838 instead of the 1000 in the paper

        self.imgs = [p for p in self.imgs if p.stem.split('_')[0] in self.author_ids]
        self.labels = [img.stem.split('_')[0] for img in self.imgs]
