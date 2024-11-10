from PIL import Image
from .base_dataset import BaseSHTGDataset, download_file, extract_zip
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
import gzip

SHTG_KARAOKE_HANDW_URL = {
    'lines': 'https://github.com/aimagelab/HWD/releases/download/karaoke/shtg_karaoke_handw_lines.json.gz',
    'words': 'https://github.com/aimagelab/HWD/releases/download/karaoke/shtg_karaoke_handw_words.json.gz'
}
SHTG_KARAOKE_HANDW_PATH = {
    'lines': Path('~/.cache/karaoke/shtg_karaoke_handw_lines.json.gz').expanduser(),
    'words': Path('~/.cache/karaoke/shtg_karaoke_handw_words.json.gz').expanduser()
}

SHTG_KARAOKE_TYPEW_URL = {
    'lines': 'https://github.com/aimagelab/HWD/releases/download/karaoke/shtg_karaoke_typew_lines.json.gz',
    'words': 'https://github.com/aimagelab/HWD/releases/download/karaoke/shtg_karaoke_typew_words.json.gz'
}
SHTG_KARAOKE_TYPEW_PATH = {
    'lines': Path('~/.cache/karaoke/shtg_karaoke_typew_lines.json.gz').expanduser(),
    'words': Path('~/.cache/karaoke/shtg_karaoke_typew_words.json.gz').expanduser()
}

KARAOKE_HANDW_URL = 'https://github.com/aimagelab/HWD/releases/download/karaoke/handwritten_images.zip'
KARAOKE_HANDW_ZIP_PATH = Path('~/.cache/karaoke/handwritten_images.zip').expanduser()
KARAOKE_HANDW_DIR_PATH = Path('~/.cache/karaoke/handwritten_images').expanduser()

KARAOKE_TYPEW_URL = 'https://github.com/aimagelab/HWD/releases/download/karaoke/typewritten_images.zip'
KARAOKE_TYPEW_ZIP_PATH = Path('~/.cache/karaoke/typewritten_images.zip').expanduser()
KARAOKE_TYPEW_DIR_PATH = Path('~/.cache/karaoke/typewritten_images').expanduser()


class KaraokeBase(BaseSHTGDataset):
    def __init__(self, img_type, flavor, **kwargs):
        super().__init__(**kwargs)

        if flavor == 'handwritten':
            url = KARAOKE_HANDW_URL
            zip_path = KARAOKE_HANDW_ZIP_PATH
            dir_path = KARAOKE_HANDW_DIR_PATH
            shtg_url = SHTG_KARAOKE_HANDW_URL[img_type]
            self.shtg_path = SHTG_KARAOKE_HANDW_PATH[img_type]
        elif flavor == 'typewritten':
            url = KARAOKE_TYPEW_URL
            zip_path = KARAOKE_TYPEW_ZIP_PATH
            dir_path = KARAOKE_TYPEW_DIR_PATH
            shtg_url = SHTG_KARAOKE_TYPEW_URL[img_type]
            self.shtg_path = SHTG_KARAOKE_TYPEW_PATH[img_type]
        else:
            raise ValueError

        if not dir_path.exists():
            download_file(url, zip_path)
            extract_zip(zip_path, dir_path.parent, delete=True)

        download_file(shtg_url, self.shtg_path, exist_ok=True)
        with gzip.open(self.shtg_path, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        imgs_path = dir_path / img_type
        self.imgs = {img_path.stem: img_path for img_path in imgs_path.rglob('*.png')}

        lables_path = imgs_path / 'transcriptions.json'
        labels = json.loads(lables_path.read_text())
        self.labels = {Path(path).stem: lbl for path, lbl in labels.items()}


    def generate_shtg_data(self):
        data = []
        for sample_id, text in tqdm(self.labels.items()):
            font = Path(self.imgs[sample_id]).parent.stem

            style_ids = []
            for sample_id_tgt, text_tgt in self.labels.items():
                font_tgt = Path(self.imgs[sample_id_tgt]).parent.stem
                if font == font_tgt and sample_id != sample_id_tgt:
                    style_ids.append(sample_id_tgt)

            data.append({
                'text': text,
                'gen_id': sample_id,
                'dst': f'{font}/{sample_id}.png',
                'style_ids': style_ids,
            })
        return data



class KaraokeWords(KaraokeBase):
    def __init__(self, flavor, **kwargs):
        super().__init__('words', flavor, **kwargs)


class KaraokeLines(KaraokeBase):
    def __init__(self, flavor, **kwargs):
        super().__init__('lines', flavor, **kwargs)


class KaraokeBlankFromBlank(KaraokeBase):
    def __init__(self, gen_type, style_type, flavor, **kwargs):
        super().__init__(gen_type, flavor, **kwargs)
        alt_db = KaraokeBase(style_type, flavor, **kwargs)
        font_to_lines = defaultdict(set)
        for sample in alt_db.data:
            font = Path(sample['dst']).parent.name
            font_to_lines[font].update(sample['style_ids'])

        for sample in self.data:
            font = Path(sample['dst']).parent.name
            word_ids = list(font_to_lines[font])
            word_ids = [f'alt_{word_id}' for word_id in word_ids]
            sample['style_ids'] = word_ids
        
        alt_db.imgs = {f'alt_{word_id}': img_path for word_id, img_path in alt_db.imgs.items()}
        self.imgs = self.imgs | alt_db.imgs
        alt_db.labels = {f'alt_{word_id}': lbl for word_id, lbl in alt_db.labels.items()}
        self.labels = self.labels | alt_db.labels


class KaraokeWordsFromLines(KaraokeBlankFromBlank):
    def __init__(self, flavor, **kwargs):
        super().__init__('words', 'lines', flavor, **kwargs)


class KaraokeLinesFromWords(KaraokeBlankFromBlank):
    def __init__(self, flavor, **kwargs):
        super().__init__('lines', 'words', flavor, **kwargs)