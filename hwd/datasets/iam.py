from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random
import xml.etree.ElementTree as ET


class IAMDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None, dataset_type='words'):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, nameset)

        self.imgs = list(Path(path, dataset_type).rglob('*.png'))
        xml_files = list(Path(path, 'xmls').rglob('*.xml'))
        author_dict = {xml_file.stem: ET.parse(xml_file).getroot().attrib['writer-id'] for xml_file in xml_files}

        self.author_ids = list(set(author_dict.values()))
        self.labels = [author_dict[img.parent.stem] for img in self.imgs]

# from PIL import Image
# from .base_dataset import BaseSHTGDataset, download_file, extract_tgz
# from pathlib import Path
# import json
# import gzip
# import xml.etree.ElementTree as ET

# SHTG_IAM_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/shtg_iam_lines.json.gz'
# SHTG_IAM_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/shtg_iam_words.json.gz'
# SHTG_IAM_LINES_PATH = Path('.cache/iam/shtg_iam_lines.json.gz')
# SHTG_IAM_WORDS_PATH = Path('.cache/iam/shtg_iam_words.json.gz')

# SHTG_AUTHORS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/gan.iam.test.gt.filter27.txt'
# SHTG_AUTHORS_PATH = Path('.cache/iam/gan.iam.test.gt.filter27.txt')

# IAM_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/lines.tgz'
# IAM_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/words.tgz'
# IAM_XML_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/xml.tgz'
# IAM_ASCII_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/ascii.tgz'

# IAM_LINES_TGZ_PATH = Path('.cache/iam/lines.tgz')
# IAM_WORDS_TGZ_PATH = Path('.cache/iam/words.tgz')
# IAM_XML_TGZ_PATH = Path('.cache/iam/xml.tgz')
# IAM_ASCII_TGZ_PATH = Path('.cache/iam/ascii.tgz')

# IAM_LINES_DIR_PATH = Path('.cache/iam/lines')
# IAM_WORDS_DIR_PATH = Path('.cache/iam/words')
# IAM_XML_DIR_PATH = Path('.cache/iam/xml')
# IAM_ASCII_DIR_PATH = Path('.cache/iam/ascii')


# def extract_lines_from_xml(xml_string):
#     # Parse the XML string
#     root = ET.fromstring(xml_string)
    
#     lines_info = []

#     # Find all line elements within the handwritten-part
#     for line in root.findall('.//line'):
#         line_data = {
#             'id': line.get('id'),
#             'text': line.get('text'),
#             'writer_id': root.attrib['writer-id']
#         }
#         lines_info.append(line_data)
    
#     return lines_info


# class IAMLines(BaseSHTGDataset):
#     def __init__(self, load_style_samples=True, num_style_samples=1):
#         self.load_style_samples = load_style_samples
#         self.num_style_samples = num_style_samples

#         download_file(SHTG_IAM_LINES_URL, SHTG_IAM_LINES_PATH, exist_ok=True)
#         download_file(SHTG_AUTHORS_URL, SHTG_AUTHORS_PATH, exist_ok=True)
#         self.authors = set()
#         for author_line in SHTG_AUTHORS_PATH.read_text().splitlines():
#             self.authors.add(author_line.split(',')[0])

#         if not IAM_LINES_DIR_PATH.exists():
#             download_file(IAM_LINES_URL, IAM_LINES_TGZ_PATH)
#             extract_tgz(IAM_LINES_TGZ_PATH, IAM_LINES_DIR_PATH, delete=True)

#         if not IAM_XML_DIR_PATH.exists():
#             download_file(IAM_XML_URL, IAM_XML_TGZ_PATH)
#             extract_tgz(IAM_XML_TGZ_PATH, IAM_XML_DIR_PATH, delete=True)

#         with gzip.open(SHTG_IAM_LINES_PATH, 'rt', encoding='utf-8') as file:
#             self.data = json.load(file)

#         self.imgs = {img_path.stem: img_path for img_path in IAM_LINES_DIR_PATH.rglob('*.png')}

#         lines = []
#         for xml_file in IAM_XML_DIR_PATH.rglob('*.xml'):
#             lines.extend(extract_lines_from_xml(xml_file.read_text()))
#         lines = [line for line in lines if line['writer_id'] in self.authors]
#         self.lines = {line['id']: line for line in lines}

#         # Switching from words ids to lines ids
#         for sample in self.data:
#             filtered_style_ids = []
#             for style_sample in sample['style_ids']:
#                 if self.lines[style_sample[:-3]]['text'] != sample['word']:
#                     filtered_style_ids.append(style_sample[:-3])
#             sample['style_ids'] = list(set(filtered_style_ids))
#             assert len(sample['style_ids']) > 0
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         output = {}
#         output['gen_text'] = sample['word']
#         output['author'] = Path(sample['dst']).parent.name
#         output['dst_path'] = sample['dst']
#         output['style_ids'] = sample['style_ids'][:self.num_style_samples]
#         output['style_imgs_path'] = [self.imgs[id] for id in output['style_ids']]
#         if self.load_style_samples:
#             output['style_imgs'] = [Image.open(self.imgs[id]) for id in output['style_ids']]
#         return output

