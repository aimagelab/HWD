from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random

class HKRDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        super().__init__(path, transform, author_ids, nameset, max_samples)

        path = Path(path)
        imgs = list(p for p in (path / 'img').iterdir())
