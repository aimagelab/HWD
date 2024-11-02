from .base_dataset import BaseDataset
from pathlib import Path


class FolderDataset(BaseDataset):
    def __init__(self, path, extension='png', **kwargs):
        imgs = list(Path(path).rglob(f'*.{extension}'))
        assert len(imgs) > 0, 'No images found.'
        authors = [img.parent.name for img in imgs]
        author_ids = sorted(set(authors))
        super(FolderDataset, self).__init__(path, imgs, authors, author_ids, **kwargs)