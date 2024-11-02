from .base_dataset import BaseDataset
from pathlib import Path
import msgpack


class WashingtonDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
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

        super().__init__(Path(path), imgs, [0] * len(imgs), [0], nameset=nameset, transform=transform)

        if nameset is not None:
            data_path = Path(path) / f'{nameset}.msgpack'
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    data = dict(msgpack.unpack(f, raw=False))
                    self.imgs = [img for img in self.imgs if img.name in data]
                    self.authors = [0] * len(self.imgs)
                    self.labels = [data[img.name] for img in self.imgs]

