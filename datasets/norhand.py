import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import csv

import base_dataset


class NorhandDataset(base_dataset.BaseDataset):
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

        if author_ids is None:
            self.author_ids = list(authors)

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
    norhand_path = r'/mnt/FoMo_AIISDH/datasets/Norhand'

    # Code to generate the authors ids file
    # imgs_paths = sorted(list(Path(norhand_path).glob('*.jpg')))
    # prefixes = [p.stem.split('_')[0] for p in imgs_paths]
    # set_prefixes = set(prefixes)
    # prefixes_dict = {i: p for i, p in enumerate(set_prefixes)}
    # import json
    # with open('/mnt/FoMo_AIISDH/datasets/Norhand/authors_ids_file.json', 'w') as authors_ids_file:
    #     json.dump(prefixes_dict, authors_ids_file)

    dataset = NorhandDataset(norhand_path)
    print(dataset)
