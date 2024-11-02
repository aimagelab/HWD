from PIL import Image
from .base_dataset import BaseSHTGDataset, download_file, extract_zip
from pathlib import Path

RIMES_URL = 'https://storage.teklia.com/public/rimes2011/RIMES-2011-Lines.zip'
RIMES_ZIP_PATH = Path('.cache/rimes/RIMES-2011-Lines.zip')
RIMES_DIR_PATH = Path('.cache/rimes')


class Rimes(BaseSHTGDataset):
    def __init__(self, load_style_samples=True, num_style_samples=1, scenario=None):
        super().__init__(load_style_samples=load_style_samples, num_style_samples=num_style_samples, scenario=scenario)

        if not RIMES_DIR_PATH.exists():
            download_file(RIMES_URL, RIMES_ZIP_PATH)
            extract_zip(RIMES_ZIP_PATH, RIMES_DIR_PATH, delete=True)

        self.imgs = {img_path.stem: img_path for img_path in RIMES_DIR_PATH.rglob('*.jpg')}
        labels_path = RIMES_DIR_PATH / 'RIMES-2011-Lines' / 'Transcriptions'
        self.labels = {lbl_path.stem: lbl_path.read_text() for lbl_path in labels_path.rglob('*.txt')}
        
        test_set = RIMES_DIR_PATH / 'RIMES-2011-Lines' / 'Sets' / 'TestLines.txt'
        test_set = test_set.read_text().splitlines()
        test_set = {sample: self._unpack_sample(sample) for sample in test_set}

        self.data = []
        for sample_id, (nameset, author, num) in test_set.items():
            style_ids = []
            for sample_id_2, (_, author_2, _) in test_set.items():
                 if author == author_2 and sample_id != sample_id_2:
                     style_ids.append(sample_id_2)
            self.data.append({
                'word': self.labels[sample_id],
                'dst': f'test/{author}/{sample_id}.png',
                'style_ids': style_ids,
            })

    def _unpack_sample(self, sample):
        nameset, sample_id = sample.split('-')
        return nameset, *sample_id.split('_')