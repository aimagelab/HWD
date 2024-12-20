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
SHTG_IAM_LINES_PATH = Path('~/.cache/iam/shtg_iam_lines.json.gz').expanduser()

SHTG_IAM_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/shtg_iam_words.json.gz'
SHTG_IAM_WORDS_PATH = Path('~/.cache/iam/shtg_iam_words.json.gz').expanduser()

SHTG_AUTHORS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/gan.iam.test.gt.filter27.txt'
SHTG_AUTHORS_PATH = Path('~/.cache/iam/gan.iam.test.gt.filter27.txt').expanduser()

IAM_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/lines.tgz'
IAM_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/words.tgz'
IAM_XML_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/xml.tgz'
IAM_ASCII_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/ascii.tgz'
IAM_FORMS_AD_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/formsA-D.tgz'
IAM_FORMS_EH_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/formsE-H.tgz'
IAM_FORMS_IZ_URL = 'https://github.com/aimagelab/HWD/releases/download/iam/formsI-Z.tgz'

IAM_LINES_TGZ_PATH = Path('~/.cache/iam/lines.tgz').expanduser()
IAM_WORDS_TGZ_PATH = Path('~/.cache/iam/words.tgz').expanduser()
IAM_XML_TGZ_PATH = Path('~/.cache/iam/xml.tgz').expanduser()
IAM_ASCII_TGZ_PATH = Path('~/.cache/iam/ascii.tgz').expanduser()
IAM_FORMS_AD_TGZ_PATH = Path('~/.cache/iam/formsA-D.tgz').expanduser()
IAM_FORMS_EH_TGZ_PATH = Path('~/.cache/iam/formsE-H.tgz').expanduser()
IAM_FORMS_IZ_TGZ_PATH = Path('~/.cache/iam/formsI-Z.tgz').expanduser()

IAM_LINES_DIR_PATH = Path('~/.cache/iam/lines').expanduser()
IAM_WORDS_DIR_PATH = Path('~/.cache/iam/words').expanduser()
IAM_XML_DIR_PATH = Path('~/.cache/iam/xml').expanduser()
IAM_ASCII_DIR_PATH = Path('~/.cache/iam/ascii').expanduser()
IAM_FORMS_AD_DIR_PATH = Path('~/.cache/iam/formsA-D').expanduser()
IAM_FORMS_EH_DIR_PATH = Path('~/.cache/iam/formsE-H').expanduser()
IAM_FORMS_IZ_DIR_PATH = Path('~/.cache/iam/formsI-Z').expanduser()

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

            if len(style_ids) == 0:
                continue

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


class IAMLinesAndWords(IAMBase):
    def __init__(self, shtg_url, shtg_path, **kwargs):
        super().__init__(**kwargs)

        download_file(shtg_url, shtg_path, exist_ok=True)

        if not IAM_WORDS_DIR_PATH.exists():
            download_file(IAM_WORDS_URL, IAM_WORDS_TGZ_PATH)
            extract_tgz(IAM_WORDS_TGZ_PATH, IAM_WORDS_DIR_PATH, delete=True)

        if not IAM_LINES_DIR_PATH.exists():
            download_file(IAM_LINES_URL, IAM_LINES_TGZ_PATH)
            extract_tgz(IAM_LINES_TGZ_PATH, IAM_LINES_DIR_PATH, delete=True)

        self.shtg_path = shtg_url
        with gzip.open(shtg_path, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        self.words_images = {img_path.stem: img_path for img_path in IAM_WORDS_DIR_PATH.rglob('*.png')}
        self.lines_images = {img_path.stem: img_path for img_path in IAM_LINES_DIR_PATH.rglob('*.png')}
        self.imgs = {}
        self.imgs.update(self.words_images)
        self.imgs.update(self.lines_images)

        self.words_labels = {word['id']: word['text'] for word in self.words}
        self.lines_labels = {line['id']: line['text'] for line in self.lines}
        self.labels = {}
        self.labels.update(self.words_labels)
        self.labels.update(self.lines_labels)


class IAMLinesFromWords(IAMLinesAndWords):
    def __init__(self, **kwargs):
        super().__init__(SHTG_IAM_LINES_URL, SHTG_IAM_LINES_PATH, **kwargs)

        lines_to_words = defaultdict(list)
        for word_id in self.words_labels.keys():
            a, b, c, _ = word_id.split('-')
            lines_to_words[f'{a}-{b}-{c}'].append(word_id)

        for sample in self.data:
            style_ids = []
            for word_id in sample['style_ids']:
                style_ids.extend(lines_to_words[word_id])
            sample['style_ids'] = style_ids


class IAMWordsFromLines(IAMLinesAndWords):
    def __init__(self, **kwargs):
        super().__init__(SHTG_IAM_WORDS_URL, SHTG_IAM_WORDS_PATH, **kwargs)

        words_to_lines = defaultdict(list)
        for word_id in self.words_labels.keys():
            a, b, c, _ = word_id.split('-')
            words_to_lines[word_id].append(f'{a}-{b}-{c}')

        for sample in self.data:
            style_ids = []
            for word_id in sample['style_ids']:
                style_ids.extend(words_to_lines[word_id])
            sample['style_ids'] = list(set(style_ids))



