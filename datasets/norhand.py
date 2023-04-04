import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import csv

from .base_dataset import BaseDataset


class NorhandDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
        """
        super().__init__(path, transform, author_ids, nameset)

        authors = set()
        self.pages_and_authors = {}
        with open('/mnt/FoMo_AIISDH/datasets/Norhand/writer_information.csv', 'r') as authors_ids_file:
            csvreader = csv.reader(authors_ids_file, delimiter=';')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                authors.add(row[2])
                self.pages_and_authors[row[0]] = row[2]

        authors = sorted(list(authors))
        if author_ids is None:
            self.author_ids = sorted(list(authors))

        self.labels = {str(author): int(label) for label, author in enumerate(authors)}
        self.imgs = [
            p for p in Path(path).rglob(f'*.jpg') if self.pages_and_authors[p.stem.split('_')[0]] in self.author_ids]

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[self.pages_and_authors[img.stem.split('_')[0]]]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    norhand_path = r'/home/shared/datasets/Norhand'

    dataset = NorhandDataset(norhand_path)
    print(len(dataset))
    print(dataset[0][1])

    dataset = NorhandDataset(norhand_path, author_ids=['Bonnevie, Kristine', 'Nielsen, Petronelle'])
    print(len(dataset))
    print(dataset[0][1])
