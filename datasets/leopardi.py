from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random



class LeopardiDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None, max_samples=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, None, nameset, max_samples)

        if author_ids is None:
            self.author_ids = [0, ]

        self.imgs = [p for p in Path(path).rglob(f'*.jpg')]
        self.labels = [0, ] * len(self.imgs)
        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]


if __name__ == '__main__':
    # leopardi_path = r'/mnt/FoMo_AIISDH/datasets/LEOPARDI/leopardi'
    leopardi_path = r'/home/shared/datasets/leopardi'

    dataset = LeopardiDataset(leopardi_path)
    print(len(dataset))
    print(dataset[0])
