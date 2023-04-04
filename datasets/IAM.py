import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import json
import random

from .base_dataset import BaseDataset


class IAMDataset(BaseDataset):
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

        all_author_ids = sorted([p.stem for p in Path(path).iterdir()])
        self.labels = {str(author_id): int(label) for label, author_id in enumerate(all_author_ids)}
        if author_ids is None:
            self.author_ids = all_author_ids

        self.imgs = [p for author_id in self.author_ids for p in Path(path).rglob(f'{author_id}*') if p.is_file()]
        if max_samples is not None:
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[img.stem.split('-')[0]]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    # iam_path = r'/mnt/beegfs/work/FoMo_AIISDH/datasets/IAM'
    iam_path = r'/home/shared/datasets/IAM'

    dataset = IAMDataset(iam_path)
    print(len(dataset))
    print(dataset[0][1])

    dataset = IAMDataset(iam_path, author_ids=['r06', 'p06'])
    print(len(dataset))
    print(dataset[0][1])
