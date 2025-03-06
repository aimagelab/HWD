from .base_dataset import BaseDataset
from pathlib import Path
from .shtg.base_dataset import download_file, extract_zip
import requests

URL_TEMPLATE = 'https://github.com/aimagelab/HWD/releases/download/generated/{}.zip'
DIR_PATH = Path('~/.cache/generated').expanduser()

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


class GeneratedDataset(FolderDataset):
    def __init__(self, name, **kwargs):
        assert '__' in name, 'The name of the dataset must be in the format "dataset__model"'
        url = URL_TEMPLATE.format(name)
        dataset, model = name.split('__')
        dir_path = DIR_PATH / dataset
        
        if not (dir_path / model).exists():
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code != 200:
                raise ValueError(f'{name} dataset not found.')
            
            download_file(url, DIR_PATH / f'{name}.zip')
            extract_zip(DIR_PATH/f'{name}.zip', dir_path, delete=True)

        super(GeneratedDataset, self).__init__(dir_path / model, **kwargs)