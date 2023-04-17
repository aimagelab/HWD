from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random
import json
import numpy as np


def extract_words(path):
    imgs = list(p for p in (Path(path) / 'raw' / 'raw').rglob('*.jpg'))
    jsons = [Path(str(p).replace('jpg', 'json')) for p in imgs]
    authors = set([p.stem.split('_')[0] for p in imgs])

    ai = 0
    save_folder = (Path(path) / 'words')
    save_folder.mkdir(exist_ok=True)
    for img_path, json_path in zip(imgs, jsons):
        print(f'[{ai}]/[{len(imgs)}] Processing {img_path}...')
        ai += 1
        author = img_path.stem.split('_')[0]
        author_folder = save_folder / author
        author_folder.mkdir(exist_ok=True)
        img = np.array(Image.open(img_path))
        with open(json_path, 'r') as f:
            data = json.load(f)
        for i, shape in enumerate(data['shapes']):
            x1 = int(np.round(shape['points'][0][0]))
            y1 = int(np.round(shape['points'][0][1]))
            x2 = int(np.round(shape['points'][1][0]))
            y2 = int(np.round(shape['points'][1][1]))
            word = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            word = Image.fromarray(word)
            word.save(Path(author_folder / f'{author}_{i}.jpg'))


class BanglaWritingDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        super().__init__(path, transform, author_ids, nameset, max_samples)

        path = Path(path) / 'words'
        self.all_author_ids = sorted([p.stem for p in path.iterdir()])
        if author_ids is None:
            self.author_ids = self.all_author_ids

        self.imgs = [p for author_id in self.author_ids for p in path.glob(f'{author_id}/*.jpg')]

        self.labels = [p.parent for p in self.imgs]

        if max_samples is not None:
            self.imgs = sorted(self.imgs)
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]
            self.labels = self.labels[:max_samples]
