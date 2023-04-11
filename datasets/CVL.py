from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class CVLDataset(BaseDataset):
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

        # self.labels = {str(author_id): int(label) for label, author_id in enumerate(self.all_author_ids)}
        if author_ids is None:
            self.all_author_ids = sorted([p.stem for p in Path(path).glob('*/lines/*')])
            self.author_ids = self.all_author_ids

        self.imgs = [p for p in Path(path).rglob(f'lines/*/*')]
        self.labels = [img.parent.stem for img in self.imgs]
        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]


if __name__ == '__main__':
    # cvl_path = r'/mnt/FoMo_AIISDH/datasets/CVL/cvl-database-1-1'
    cvl_path = r'/home/shared/datasets/cvl-database-1-1'

    dataset = CVLDataset(cvl_path)
    print(len(dataset))
    print(dataset[0][1])

    dataset = CVLDataset(cvl_path, author_ids=['0953', '0479'])
    print(len(dataset))
    print(dataset[0][1])
