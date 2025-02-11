from .base_dataset import BaseDataset
from pathlib import Path
import msgpack


class RimesDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        raise NotImplementedError
        """
               Args:
                   path (string): Path folder of the dataset.
                   transform (callable, optional): Optional transform to be applied
                       on a sample.
                   author_ids (list, optional): List of authors to consider.
                   nameset (string, optional): Name of the dataset.
                   max_samples (int, optional): Maximum number of samples to consider.
               """

        imgs_dir = Path(path) / 'lines'
        imgs = list(imgs_dir.rglob('*.png'))

        super().__init__(path, transform, nameset)

        self.author_ids = {img.stem.split('-')[1] for img in self.imgs}

        self.imgs = [img for img in self.imgs if img.stem.split('-')[1] in self.author_ids]
        self.author_ids = sorted(self.author_ids)
        self.labels = [img.stem.split('-')[1] for img in self.imgs]
