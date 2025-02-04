from .base_dataset import BaseDataset
from pathlib import Path
from .shtg.base_dataset import extract_zip, extract_tgz
from .shtg.iam import download_file, extract_lines_from_xml, extract_words_from_xml
from .shtg.iam import IAM_XML_DIR_PATH, IAM_XML_TGZ_PATH, IAM_XML_URL
from .shtg.iam import IAM_WORDS_DIR_PATH, IAM_WORDS_TGZ_PATH, IAM_WORDS_URL
from .shtg.iam import IAM_LINES_DIR_PATH, IAM_LINES_TGZ_PATH, IAM_LINES_URL
from .shtg.iam import SHTG_AUTHORS_URL, SHTG_AUTHORS_PATH

IAM_SPLITS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/largeWriterIndependentTextLineRecognitionTask.1.zip'
IAM_SPLITS_ZIP_PATH = Path('~/.cache/iam/largeWriterIndependentTextLineRecognitionTask.1.zip').expanduser()
IAM_SPLITS_DIR_PATH = Path('~/.cache/iam/largeWriterIndependentTextLineRecognitionTask.1').expanduser()

IAM_SPLITS_PATHS = {
    'train': IAM_SPLITS_DIR_PATH / 'trainset.txt',
    'test': IAM_SPLITS_DIR_PATH / 'testset.txt',
    'val1': IAM_SPLITS_DIR_PATH / 'validationset1.txt',
    'val2': IAM_SPLITS_DIR_PATH / 'validationset2.txt',
}


class IAMDataset(BaseDataset):
    def __init__(self, transform=None, nameset='train', dataset_type='words', split_type='shtg'):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """

        if not IAM_XML_DIR_PATH.exists():
            download_file(IAM_XML_URL, IAM_XML_TGZ_PATH)
            extract_tgz(IAM_XML_TGZ_PATH, IAM_XML_DIR_PATH, delete=True)

        if dataset_type == 'lines':
            self.data = []
            for xml_file in IAM_XML_DIR_PATH.rglob('*.xml'):
                self.data.extend(extract_lines_from_xml(xml_file.read_text()))

            if not IAM_LINES_DIR_PATH.exists():
                download_file(IAM_LINES_URL, IAM_LINES_TGZ_PATH)
                extract_tgz(IAM_LINES_TGZ_PATH, IAM_LINES_DIR_PATH, delete=True)

            self.dict_path = {img_path.stem: img_path for img_path in IAM_LINES_DIR_PATH.rglob('*.png')}

        else:
            self.data = []
            for xml_file in IAM_XML_DIR_PATH.rglob('*.xml'):
                self.data.extend(extract_words_from_xml(xml_file.read_text()))

            if not IAM_WORDS_DIR_PATH.exists():
                download_file(IAM_WORDS_URL, IAM_WORDS_TGZ_PATH)
                extract_tgz(IAM_WORDS_TGZ_PATH, IAM_WORDS_DIR_PATH, delete=True)

            self.dict_path = {img_path.stem: img_path for img_path in IAM_WORDS_DIR_PATH.rglob('*.png')}

        if split_type == 'shtg':
            assert nameset in ['train', 'test'], f"Invalid nameset: {nameset}. Available namesets: ['train', 'test']"

            if not SHTG_AUTHORS_PATH.exists():
                download_file(SHTG_AUTHORS_URL, SHTG_AUTHORS_PATH, exist_ok=True)
            
            test_authors = SHTG_AUTHORS_PATH.read_text().splitlines()
            test_authors = {line.split(',')[0] for line in test_authors}

            if nameset == 'train':
                self.data = [sample for sample in self.data if sample['writer_id'] not in test_authors]
            else:
                self.data = [sample for sample in self.data if sample['writer_id'] in test_authors]

        elif split_type == 'original':
            assert nameset in IAM_SPLITS_PATHS, f"Invalid nameset: {nameset}. Available namesets: {list(IAM_SPLITS_PATHS.keys())}"

            if not IAM_SPLITS_DIR_PATH.exists():
                download_file(IAM_SPLITS_URL, IAM_SPLITS_ZIP_PATH)
                extract_zip(IAM_SPLITS_ZIP_PATH, IAM_SPLITS_DIR_PATH, delete=True)
            
            allowed_ids = set(IAM_SPLITS_PATHS[nameset].read_text().splitlines())

            def _id_line(id):
                return id[:-3] if dataset_type == 'words' else id
            self.data = [sample for sample in self.data if _id_line(sample['id']) in allowed_ids]
        else:
            raise ValueError(f"Invalid split type: {split_type}. Available split types: ['shtg', 'original']")

        self.imgs = [self.dict_path[sample['id']] for sample in self.data]
        self.author_ids = [sample['writer_id'] for sample in self.data]
        super().__init__(IAM_XML_DIR_PATH.parent, self.imgs, self.data, self.author_ids, nameset, transform=transform)

        self.labels = [sample['text'] for sample in self.data]
        self.has_labels = True
