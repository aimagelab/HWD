from .base_dataset import BaseDataset
from pathlib import Path
from .shtg.base_dataset import extract_zip, download_file

WASHINGTON_URL = 'https://github.com/aimagelab/HWD/releases/download/washington/washingtondb-v1.0.zip'
WASHINGTON_ZIP_PATH = Path('~/.cache/washington/washingtondb-v1.0.zip').expanduser()
WASHINGTON_DIR_PATH = Path('~/.cache/washington').expanduser()

SPECIAL_MAP = {
    "s_0": "0", "s_1": "1", "s_2": "2", "s_3": "3", "s_4": "4",
    "s_5": "5", "s_6": "6", "s_7": "7", "s_8": "8", "s_9": "9",
    "s_0th": "0th", "s_1st": "1st", "s_2nd": "2nd", "s_3rd": "3rd",
    "s_4th": "4th", "s_5th": "5th", "s_6th": "6th", "s_7th": "7th",
    "s_8th": "8th", "s_9th": "9th", "s_1th": "1th",
    "s_pt": ".",  "s_cm": ",", "s_s": "S", "s_mi": "_",
    "s_sq": "Sq", "s_qt": "'", "s_GW": "G.W.", "s_qo": ":",
    "s_et": "&", "s_bl": "(", "s_br": ")", "s_lb": "L", "s_sl": "Sl",
}

class WashingtonDataset(BaseDataset):
    def __init__(self, transform=None, nameset='train', split='cv1', dataset_type='lines'):
        
        if not WASHINGTON_DIR_PATH.exists():
            download_file(WASHINGTON_URL, WASHINGTON_ZIP_PATH)
            extract_zip(WASHINGTON_ZIP_PATH, WASHINGTON_DIR_PATH, delete=True)
        washington_unzip_path = WASHINGTON_DIR_PATH / 'washingtondb-v1.0'
        
        filenames = list(washington_unzip_path.rglob(f'sets/{split}/{nameset}.txt'))
        assert len(filenames) > 0, f'No file found for {nameset} in {washington_unzip_path}'
        
        split_ids = []
        for filename in filenames:
            split_ids.extend(filename.read_text().splitlines())
        split_ids = set(split_ids)

        if dataset_type == 'lines':
            imgs_root = washington_unzip_path / 'data' / 'line_images_normalized'
            labels_path = washington_unzip_path / 'ground_truth' / 'transcription.txt'
        elif dataset_type == 'words':
            imgs_root = washington_unzip_path / 'data' / 'word_images_normalized'
            labels_path = washington_unzip_path / 'ground_truth' / 'word_labels.txt'
        else:
            raise ValueError(f'Invalid dataset_type: {dataset_type}. Available types: ["lines", "words"]')
        
        self.imgs = list(imgs_root.rglob('*.png'))
        self.imgs = [img for img in self.imgs if img.stem[:6] in split_ids]
        assert len(self.imgs) > 0, f'No images found for {nameset} in {imgs_root}'

        self.authors = ['0'] * len(self.imgs)  # All samples are from the same author
        super().__init__(
            washington_unzip_path,
            self.imgs,
            self.authors,
            ['0'],  # All samples are from the same author
            transform=transform,
        )

        self.labels_dict = {}
        self.charset = set()
        for line in labels_path.read_text().splitlines():
            img_id, label = line.split()
            label = label.replace('|', '- -').split('-')
            for i in range(len(label)):
                if len(label[i]) > 1:
                    label[i] = SPECIAL_MAP[label[i]]
            label = ''.join(label)
            self.labels_dict[img_id] = label
        self.labels = [self.labels_dict[img.stem] for img in self.imgs]
        self.has_labels = True

