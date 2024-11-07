from PIL import Image
from .base_dataset import BaseSHTGDataset, download_file, extract_tgz
from pathlib import Path
import json
import gzip
import xml.etree.ElementTree as ET
from tqdm import tqdm
import html
from collections import defaultdict

SHTG_IAM_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/shtg_iam_lines.json.gz'
SHTG_IAM_LINES_PATH = Path('.cache/iam/shtg_iam_lines.json.gz')

SHTG_IAM_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/shtg_iam_words.json.gz'
SHTG_IAM_WORDS_PATH = Path('.cache/iam/shtg_iam_words.json.gz')

SHTG_AUTHORS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/gan.iam.test.gt.filter27.txt'
SHTG_AUTHORS_PATH = Path('.cache/iam/gan.iam.test.gt.filter27.txt')

IAM_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/lines.tgz'
IAM_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/words.tgz'
IAM_XML_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/xml.tgz'
IAM_ASCII_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/ascii.tgz'

IAM_LINES_TGZ_PATH = Path('.cache/iam/lines.tgz')
IAM_WORDS_TGZ_PATH = Path('.cache/iam/words.tgz')
IAM_XML_TGZ_PATH = Path('.cache/iam/xml.tgz')
IAM_ASCII_TGZ_PATH = Path('.cache/iam/ascii.tgz')

IAM_LINES_DIR_PATH = Path('.cache/iam/lines')
IAM_WORDS_DIR_PATH = Path('.cache/iam/words')
IAM_XML_DIR_PATH = Path('.cache/iam/xml')
IAM_ASCII_DIR_PATH = Path('.cache/iam/ascii')


def extract_lines_from_xml(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)
    lines_info = []

    # Find all line elements within the handwritten-part
    for line in root.findall('.//line'):
        line_data = {
            'id': line.get('id'),
            'text': html.unescape(line.get('text')),
            'writer_id': root.attrib['writer-id']
        }
        lines_info.append(line_data)
    return lines_info


def extract_words_from_xml(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)
    words_info = []

    # Find all words elements within the handwritten-part
    for line in root.findall('.//line'):
        for word in line.findall('word'):
            word_info = {
                'id': word.get('id'),
                'text': html.unescape(word.get('text')),
                'writer_id': root.attrib['writer-id'],
            }
            words_info.append(word_info)
    return words_info


class IAMBase(BaseSHTGDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not IAM_XML_DIR_PATH.exists():
            download_file(IAM_XML_URL, IAM_XML_TGZ_PATH)
            extract_tgz(IAM_XML_TGZ_PATH, IAM_XML_DIR_PATH, delete=True)

        self.lines = []
        for xml_file in IAM_XML_DIR_PATH.rglob('*.xml'):
            self.lines.extend(extract_lines_from_xml(xml_file.read_text()))

        self.words = []
        for xml_file in IAM_XML_DIR_PATH.rglob('*.xml'):
            self.words.extend(extract_words_from_xml(xml_file.read_text()))

        self.authors = {}
        for line in self.lines:
            self.authors[line['id']] = line['writer_id']
        for word in self.words:
            self.authors[word['id']] = word['writer_id']

        download_file(SHTG_AUTHORS_URL, SHTG_AUTHORS_PATH, exist_ok=True)
        self.shtg_authors = SHTG_AUTHORS_PATH.read_text().splitlines()
        self.shtg_authors = {line.split(',')[0] for line in self.shtg_authors}

    
    def generate_shtg_data(self):
        data = []
        for sample_id, text in tqdm(self.labels.items()):
            writer_n = self.authors[sample_id]

            if not writer_n in self.shtg_authors or len(text) < 3:
                continue

            style_ids = []
            for sample_id_tgt, text_tgt in self.labels.items():
                writer_n_tgt = self.authors[sample_id_tgt]
                if writer_n == writer_n_tgt and sample_id != sample_id_tgt \
                    and len(text_tgt) > 2 and text_tgt != text:
                    style_ids.append(sample_id_tgt)

            data.append({
                'text': text,
                'gen_id': sample_id,
                'dst': f'{writer_n}/{sample_id}.png',
                'style_ids': style_ids,
            })
        return data


class IAMLines(IAMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        download_file(SHTG_IAM_LINES_URL, SHTG_IAM_LINES_PATH, exist_ok=True)

        if not IAM_LINES_DIR_PATH.exists():
            download_file(IAM_LINES_URL, IAM_LINES_TGZ_PATH)
            extract_tgz(IAM_LINES_TGZ_PATH, IAM_LINES_DIR_PATH, delete=True)

        self.imgs = {img_path.stem: img_path for img_path in IAM_LINES_DIR_PATH.rglob('*.png')}
        self.labels = {line['id']: line['text'] for line in self.lines}

        self.shtg_path = SHTG_IAM_LINES_PATH
        with gzip.open(SHTG_IAM_LINES_PATH, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)


class IAMWords(IAMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        download_file(SHTG_IAM_WORDS_URL, SHTG_IAM_WORDS_PATH, exist_ok=True)

        if not IAM_WORDS_DIR_PATH.exists():
            download_file(IAM_WORDS_URL, IAM_WORDS_TGZ_PATH)
            extract_tgz(IAM_WORDS_TGZ_PATH, IAM_WORDS_DIR_PATH, delete=True)

        self.shtg_path = SHTG_IAM_WORDS_PATH
        with gzip.open(SHTG_IAM_WORDS_PATH, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        self.imgs = {img_path.stem: img_path for img_path in IAM_WORDS_DIR_PATH.rglob('*.png')}
        self.labels = {word['id']: word['text'] for word in self.words}


class IAMLinesFromWords(IAMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        download_file(SHTG_IAM_LINES_URL, SHTG_IAM_LINES_PATH, exist_ok=True)

        if not IAM_WORDS_DIR_PATH.exists():
            download_file(IAM_WORDS_URL, IAM_WORDS_TGZ_PATH)
            extract_tgz(IAM_WORDS_TGZ_PATH, IAM_WORDS_DIR_PATH, delete=True)

        if not IAM_LINES_DIR_PATH.exists():
            download_file(IAM_LINES_URL, IAM_LINES_TGZ_PATH)
            extract_tgz(IAM_LINES_TGZ_PATH, IAM_LINES_DIR_PATH, delete=True)

        self.shtg_path = SHTG_IAM_LINES_PATH
        with gzip.open(SHTG_IAM_LINES_PATH, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        words_images = {img_path.stem: img_path for img_path in IAM_WORDS_DIR_PATH.rglob('*.png')}
        lines_images = {img_path.stem: img_path for img_path in IAM_LINES_DIR_PATH.rglob('*.png')}
        self.imgs = words_images | lines_images

        words_lables = {word['id']: word['text'] for word in self.words}
        lines_lables = {line['id']: line['text'] for line in self.lines}
        self.labels = words_lables | lines_lables

        lines_to_words = defaultdict(list)
        for word_id in words_lables.keys():
            a, b, c, _ = word_id.split('-')
            lines_to_words[f'{a}-{b}-{c}'].append(word_id)

        for sample in self.data:
            style_ids = []
            for word_id in sample['style_ids']:
                style_ids.extend(lines_to_words[word_id])
            sample['style_ids'] = style_ids



