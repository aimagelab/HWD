from .base_dataset import BaseDataset
from pathlib import Path
import msgpack
from .shtg.base_dataset import extract_zip, download_file

LEOPARDI_URL = 'https://github.com/aimagelab/HWD/releases/download/leopardi/leopardi.zip'
LEOPARDI_ZIP_PATH = Path('~/.cache/leopardi/leopardi.zip').expanduser()
LEOPARDI_DIR_PATH = Path('~/.cache/leopardi').expanduser()


class LeopardiDataset(BaseDataset):
    def __init__(self, transform=None, nameset='train', dataset_type='lines'):
        if not LEOPARDI_DIR_PATH.exists():
            download_file(LEOPARDI_URL, LEOPARDI_ZIP_PATH)
            extract_zip(LEOPARDI_ZIP_PATH, LEOPARDI_DIR_PATH, delete=True)
        leopardi_unzip_path = LEOPARDI_DIR_PATH / 'leopardi'

        nameset_path = leopardi_unzip_path / f'{nameset}.msgpack'
        assert nameset_path.exists(), f'The nameset file {nameset_path} does not exist at the specified path {nameset_path}'
        
        with open(nameset_path, 'rb') as f:
            data = msgpack.load(f)

        if dataset_type == 'lines':
            self.path = leopardi_unzip_path / 'lines'
            img_suffix = '.jpg'
        elif dataset_type == 'lines_binarized':
            self.path = leopardi_unzip_path / 'binarized_lines'
            img_suffix = '.png'
        else:
            raise ValueError(f'Invalid dataset_type: {dataset_type}. Available types: ["lines", "lines_binarized"]')
        
        self.imgs = [self.path / img for img, _ in data]
        self.imgs = [img_path.with_suffix(img_suffix) for img_path in self.imgs]
        for img_path in self.imgs:
            assert img_path.exists(), f'Image {img_path} does not exist'
        
        self.author_ids = [0, ] * len(self.imgs)  # All samples are from the same author
        super().__init__(
            leopardi_unzip_path,
            self.imgs,
            self.author_ids,
            [0, ],  # All samples are from the same author
            transform=transform,
        )
        
        self.labels = [label for _, label in data]
        self.has_labels = True

