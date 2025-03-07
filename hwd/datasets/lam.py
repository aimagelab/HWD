import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import json
import random

from . import BaseDataset


class LAMDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        raise NotImplementedError
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): Decade to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, nameset)

        self.imgs = [p for p in Path(path).glob(f'img/*.jpg')]
        decades_json_paths = list((Path(path) / 'split' / 'decades_vs_decade').iterdir())

        self.author_ids = sorted([int(path.stem.split('_')[-1]) for path in decades_json_paths])

        images_decades = {}
        for decade_json_path in decades_json_paths:
            with open(decade_json_path, 'r') as f:
                decades = json.load(f)
            for el in decades:
                images_decades[el['img']] = el['decade_id']

        self.labels = [images_decades[img.name] for img in self.imgs]
