from PIL import Image
from .base_dataset import BaseSHTGDataset, download_file, extract_zip
from pathlib import Path
from tqdm import tqdm
import json
import gzip


SHTG_RIMES_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/rimes/shtg_rimes_lines.json.gz'
SHTG_RIMES_LINES_PATH = Path('.cache/rimes/shtg_rimes_lines.json.gz')

RIMES_URL = 'https://storage.teklia.com/public/rimes2011/RIMES-2011-Lines.zip'
RIMES_ZIP_PATH = Path('.cache/rimes/RIMES-2011-Lines.zip')
RIMES_DIR_PATH = Path('.cache/rimes')


class RimesLines(BaseSHTGDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not RIMES_DIR_PATH.exists():
            download_file(RIMES_URL, RIMES_ZIP_PATH)
            extract_zip(RIMES_ZIP_PATH, RIMES_DIR_PATH, delete=True)

        self.imgs = {img_path.stem: img_path for img_path in RIMES_DIR_PATH.rglob('*.jpg')}
        labels_path = RIMES_DIR_PATH / 'RIMES-2011-Lines' / 'Transcriptions'
        self.labels = {lbl_path.stem: lbl_path.read_text() for lbl_path in labels_path.rglob('*.txt')}

        self.shtg_path = SHTG_RIMES_LINES_PATH
        download_file(SHTG_RIMES_LINES_URL, SHTG_RIMES_LINES_PATH, exist_ok=True)
        with gzip.open(SHTG_RIMES_LINES_PATH, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)


    def generate_shtg_data(self):
        test_set = RIMES_DIR_PATH / 'RIMES-2011-Lines' / 'Sets' / 'TestLines.txt'
        test_set = test_set.read_text().splitlines()
        test_set = {sample: self._unpack_sample(sample) for sample in test_set}

        data = []
        for sample_id, (nameset, author, num) in tqdm(test_set.items()):
            style_ids = []
            for sample_id_2, (_, author_2, _) in test_set.items():
                 if author == author_2 and sample_id != sample_id_2:
                     style_ids.append(sample_id_2)
            data.append({
                'text': self.labels[sample_id],
                'gen_id': sample_id,
                'dst': f'{author}/{sample_id}.png',
                'style_ids': style_ids,
            })
        return data


    def _unpack_sample(self, sample):
        nameset, sample_id = sample.split('-')
        return nameset, *sample_id.split('_')