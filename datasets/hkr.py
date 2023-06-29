from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random

class HKRDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        super().__init__(path, transform, nameset)
        raise NotImplementedError
