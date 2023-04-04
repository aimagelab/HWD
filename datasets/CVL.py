import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import json
import random

import base_dataset


class CVLDataset(base_dataset.BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, author_ids, nameset, max_samples)

        all_author_ids = [p.stem for p in Path(path).glob('*/lines/*')]
        self.labels = {str(author_id): int(label) for label, author_id in enumerate(all_author_ids)}
        if author_ids is None:
            self.author_ids = all_author_ids

        self.imgs = [p for author_id in self.author_ids for p in Path(path).rglob(f'lines/{author_id}/*')]
        if max_samples is not None:
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[img.parent.stem]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    # cvl_path = r'/mnt/FoMo_AIISDH/datasets/CVL/cvl-database-1-1'
    cvl_path = r'/home/shared/datasets/cvl-database-1-1'

    dataset = CVLDataset(cvl_path, max_samples=1000)
    print(len(dataset))
    print(dataset[0])

    # dataset = CVLDataset(cvl_path, author_ids=['0953', '0479'])
    # print(len(dataset))
