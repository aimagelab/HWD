from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class CHSDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        super().__init__(path, transform, nameset)

        path = Path(path)
        self.author_ids = sorted([p.stem for p in (path / 'data' / 'data' / 'test').iterdir()])

        self.imgs = [p for author_id in self.author_ids for p in (path / 'data' / 'data').rglob(f'{author_id}/*.jpg')]
        self.labels = [p.parent.stem for p in self.imgs]
