from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random


class FolderDataset(BaseDataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        super(FolderDataset, self).__init__(path, transform, author_ids, nameset, max_samples)

        self.path = Path(path)
        all_author_ids = {int(author_id.stem): author_id.stem for author_id in self.path.iterdir() if
                          author_id.is_dir()}

        authors_ids = all_author_ids.keys() if author_ids is None else author_ids
        authors_ids = [all_author_ids[author_id] for author_id in authors_ids]

        self.imgs = sum([list((self.path / str(author_id)).iterdir()) for author_id in authors_ids], [])
        if max_samples is not None:
            random.shuffle(self.imgs)
            self.imgs = self.imgs[:max_samples]

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
