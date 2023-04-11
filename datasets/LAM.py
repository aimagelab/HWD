import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import json
import random

from . import BaseDataset


class LAMDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): Decade to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, None, nameset, max_samples)

        self.imgs = [p for p in Path(path).glob(f'img/*.jpg')]
        decades_json_paths = list((Path(path) / 'split' / 'decades_vs_decade').iterdir())

        self.all_author_ids = sorted([int(path.stem.split('_')[-1]) for path in decades_json_paths])
        if author_ids is None:
            self.author_ids = self.all_author_ids
        else:
            self.author_ids = [int(author_id) for author_id in author_ids]

        images_decades = {}
        for decade_json_path in decades_json_paths:
            with open(decade_json_path, 'r') as f:
                decades = json.load(f)
            for el in decades:
                images_decades[el['img']] = el['decade_id']

        self.labels = [images_decades[img.name] for img in self.imgs]

        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]


if __name__ == '__main__':
    lam_path = r'/home/shared/datasets/LAM'

    dataset = LAMDataset(lam_path)
    print(len(dataset))
    print(dataset[0][1])

    # dataset = LAMDataset(lam_path, author_ids=['r06', 'p06'])
    # print(len(dataset))
    # print(dataset[0][1])