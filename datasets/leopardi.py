import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import json
import random

import base_dataset


class LeopardiDataset(base_dataset.BaseDataset):
    def __init__(self, path, transform=None, nameset=None, max_samples=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, None, nameset, max_samples)

        self.imgs = [p for p in Path(path).rglob(f'lines/*.jpg')]
        if max_samples is not None:
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


if __name__ == '__main__':
    # leopardi_path = r'/mnt/FoMo_AIISDH/datasets/LEOPARDI/leopardi'
    leopardi_path = r'/home/shared/datasets/LEOPARDI/leopardi'

    dataset = LeopardiDataset(leopardi_path, max_samples=1000)
    print(len(dataset))
    print(dataset[0])
