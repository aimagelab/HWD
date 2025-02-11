from .base_dataset import BaseDataset
from pathlib import Path
from .shtg.base_dataset import extract_zip, download_file


SAINTGALL_URL = 'https://github.com/aimagelab/HWD/releases/download/saintgall/saintgalldb-v1.0.zip'
SAINTGALL_ZIP_PATH = Path('~/.cache/saintgall/saintgalldb-v1.0.zip').expanduser()
SAINTGALL_DIR_PATH = Path('~/.cache/saintgall').expanduser()

SPECIAL_MAP = {
    "pt": ".",
    "et": "&",
}

class SaintGallDataset(BaseDataset):
    def __init__(self, transform=None, nameset='train', dataset_type='lines'):
        if not SAINTGALL_DIR_PATH.exists():
            download_file(SAINTGALL_URL, SAINTGALL_ZIP_PATH)
            extract_zip(SAINTGALL_ZIP_PATH, SAINTGALL_DIR_PATH, delete=True)
        saintgall_unzip_path = SAINTGALL_DIR_PATH / 'saintgalldb-v1.0'

        nameset_path = saintgall_unzip_path / 'sets' / f'{nameset}.txt'
        assert nameset_path.exists(), f'The nameset file {nameset_path} does not exist at the specified path {nameset_path}'
        split_ids = set(nameset_path.read_text().splitlines())

        if dataset_type == 'lines':
            self.path = saintgall_unzip_path / 'data' / 'line_images'
        elif dataset_type == 'lines_normalized':
            self.path = saintgall_unzip_path / 'data' / 'line_images_normalized'
        else:
            raise ValueError(f'Invalid dataset_type: {dataset_type}. Available types: ["lines", "lines_normalized"]')
        
        self.imgs = list(self.path.rglob('*.png'))
        self.imgs = [img for img in self.imgs if img.stem[:10] in split_ids]
        
        self.author_ids = [0, ] * len(self.imgs)  # All samples are from the same author
        super().__init__(
            saintgall_unzip_path,
            self.imgs,
            self.author_ids,
            [0, ],  # All samples are from the same author
            transform=transform,
        )

        labels_path = saintgall_unzip_path / 'ground_truth' / 'transcription.txt'
        self.labels_dict = {}
        for line in labels_path.read_text().splitlines():
            img_id, label, _ = line.split(' ')
            label = label.replace('|', '- -').split('-')
            for i in range(len(label)):
                if len(label[i]) > 1:
                    label[i] = SPECIAL_MAP[label[i]]
            label = ''.join(label)
            self.labels_dict[img_id] = label
        
        self.labels = [self.labels_dict[img.stem] for img in self.imgs]
        self.has_labels = True
            