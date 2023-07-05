from .base_dataset import BaseDataset
from pathlib import Path


class FolderDataset(BaseDataset):
    def __init__(self, path, extension='png', **kwargs):
        super(FolderDataset, self).__init__(path, **kwargs)
        self.path = Path(path)
        self.imgs = list(Path(path).rglob(f'*.{extension}'))
        assert len(self.imgs) > 0, 'No images found.'
        self.labels = [img.parent.name for img in self.imgs]
        self.author_ids = sorted(set(self.labels))
