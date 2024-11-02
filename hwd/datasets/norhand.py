from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import csv
import random


class NorhandDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
        """
        super().__init__(path, transform, nameset)

        authors = set()
        pages_and_authors = {}

        writer_information_path = Path(path) / 'writer_information.csv'
        with open(writer_information_path, 'r') as authors_ids_file:
            csvreader = csv.reader(authors_ids_file, delimiter=';')
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                authors.add(row[2])
                pages_and_authors[row[0]] = row[2]

        self.author_ids = sorted(list(authors))

        # self.labels = {str(author): int(label) for label, author in enumerate(authors)}
        self.imgs = [p for p in Path(path).rglob(f'*.jpg')]
        self.labels = [pages_and_authors[img.stem.split('_')[0]] for img in self.imgs]
