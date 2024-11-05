from .base_dataset import BaseDataset
from pathlib import Path


class FolderDataset(BaseDataset):
    def __init__(self, path, extension=None, **kwargs):
        imgs_ext = {
            ".bmp", ".dib", ".dcx", ".eps", ".ps", ".gif",
            ".icns", ".ico", ".im", ".jpeg", ".jpg", ".j2k",
            ".j2p", ".jpx", ".msp", ".pcx", ".png", ".pbm",
            ".pgm", ".ppm", ".pnm", ".sgi", ".spider",
            ".tiff", ".tif", ".webp",".xbm"
        } if extension is None else {extension}

        imgs = [img for img in Path(path).rglob('*') if img.is_file() and img.suffix in imgs_ext]
        assert len(imgs) > 0, 'No images found.'
        
        authors = [img.parent.name for img in imgs]
        author_ids = sorted(set(authors))
        super(FolderDataset, self).__init__(path, imgs, authors, author_ids, **kwargs)