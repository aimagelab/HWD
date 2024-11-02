from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random
import json
import numpy as np


def extract_words(path):
    imgs = list(p for p in (Path(path) / 'raw' / 'raw').rglob('*.jpg'))
    jsons = [Path(str(p).replace('jpg', 'json')) for p in imgs]

    a_idx = 0
    save_folder = (Path(path) / 'words')
    save_folder.mkdir(exist_ok=True)
    for img_path, json_path in zip(imgs, jsons):
        print(f'[{a_idx}]/[{len(imgs)}] Processing {img_path}...')
        a_idx += 1
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
    def __init__(self, path, transform=None):
        super().__init__(path, transform, None)

        path = Path(path) / 'words'
        self.imgs = list(path.rglob('*.jpg'))
        self.labels = [img.stem.split('_')[0] for img in self.imgs]
        self.author_ids = sorted(set(self.labels))
