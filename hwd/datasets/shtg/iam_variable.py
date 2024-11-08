from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import random
from tqdm import tqdm
import numpy as np
import html
import json
import editdistance
import gzip
import shutil
from .iam import IAMBase, IAM_XML_DIR_PATH
from .iam import IAM_FORMS_AD_URL, IAM_FORMS_AD_DIR_PATH, IAM_FORMS_AD_TGZ_PATH
from .iam import IAM_FORMS_EH_URL, IAM_FORMS_EH_DIR_PATH, IAM_FORMS_EH_TGZ_PATH
from .iam import IAM_FORMS_IZ_URL, IAM_FORMS_IZ_DIR_PATH, IAM_FORMS_IZ_TGZ_PATH
from .base_dataset import download_file, extract_tgz

OUTPUT_DIR = Path('.cache/iam_variable/')


def random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return "#{:02X}{:02X}{:02X}".format(red, green, blue)


class Coords:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def size(self):
        return self.height, self.width

    @property
    def shape(self):
        return self.x1, self.y1, self.x2, self.y2

    def width_scaled(self, height):
        return int(self.width * height / self.height)

    def __add__(self, other):
        return Coords(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2)
        )


def compute_coords(cmp_el):
    x = int(cmp_el.attrib['x'])
    y = int(cmp_el.attrib['y'])
    w = int(cmp_el.attrib['width'])
    h = int(cmp_el.attrib['height'])
    return Coords(x, y, x + w, y + h)


def get_word_coords(word_el):
    coords = [compute_coords(cmp_el) for cmp_el in word_el.iterfind('cmp')]
    if len(coords) == 0:
        return None
    return sum(coords, coords[0])


def indices(text, sub):
    start = 0
    res = []
    while True:
        start = text.find(sub, start)
        if start == -1:
            return res
        res.append(start)
        start += len(sub)


def string_insert(text, idx, sub):
    return text[:idx] + sub + text[idx:]


def search(text, sub):
    for i in range(1, len(sub)):
        word = string_insert(sub, i, ' ')
        if word in text:
            return indices(text, word), word
    return [], sub


def join_words(words, ref):
    start_indices = indices(ref, words[0])
    if len(start_indices) == 0:
        start_indices, word = search(ref, words[0])
        assert len(start_indices) > 0
        words[0] = word
    end_indices = indices(ref, words[-1])
    if len(end_indices) == 0:
        end_indices, word = search(ref, words[-1])
        assert len(end_indices) > 0
        words[-1] = word

    best_text = None
    best_score = 10**10
    no_space_words = ''.join(words).replace(' ', '')

    for start_idx in start_indices:
        for end_idx in end_indices:
            text = ref[start_idx:end_idx + len(words[-1])]
            no_space_text = text.replace(' ', '')
            if no_space_text == no_space_words:
                return text, 0
            dist = editdistance.eval(no_space_text, no_space_words)
            if dist < best_score:
                best_score = dist
                best_text = text
    if best_text is not None:
        return best_text, best_score
    raise ValueError('Cannot find text')


def extract_lines_and_words_from_xml(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)
    lines_info = []

    # Find all line elements within the handwritten-part
    for line in root.findall('.//line'):
        line_data = {
            'id': line.get('id'),
            'element': line,
            'text': html.unescape(line.get('text')),
            'words': [{
                'id': word.get('id'),
                'element': word,
                'text': html.unescape(word.get('text')),
            } for word in line.findall('word')]
        }
        lines_info.append(line_data)
    return root.attrib['id'], root.attrib['writer-id'], lines_info

class IAMLinesVariable(IAMBase):
    def __init__(self, min_width, max_width, height=64, **kwargs):
        super().__init__(**kwargs)

        assert min_width < max_width, f'Min width should be lower than max width {min_width=} {max_width=}'
        self.output_dir = OUTPUT_DIR / f'{min_width}_{max_width}_H{height}'
        self.shtg_path = self.output_dir.with_suffix('.json.gz')
        lables_path = self.output_dir / 'transcriptions.json'

        if not self.output_dir.exists() or not self.shtg_path.exists() or not lables_path.exists():
            if self.output_dir.exists():
                print(f'Found a incomplete folder {self.output_dir}')
                shutil.rmtree(self.output_dir)
            self.download_forms()
            self.generate_variant(min_width, max_width, height)
        else:
            with gzip.open(self.shtg_path, 'rt', encoding='utf-8') as file:
                self.data = json.load(file)

            self.imgs = {img_path.stem: img_path for img_path in self.output_dir.rglob('*.png')}
            self.labels = {Path(img_path).stem: lbl for img_path, lbl in json.loads(lables_path.read_text()).items()}


    def generate_variant(self, min_width, max_width, height):
        forms_ad = {form_path.stem: form_path for form_path in IAM_FORMS_AD_DIR_PATH.rglob('*.png')}
        forms_eh = {form_path.stem: form_path for form_path in IAM_FORMS_EH_DIR_PATH.rglob('*.png')}
        forms_iz = {form_path.stem: form_path for form_path in IAM_FORMS_IZ_DIR_PATH.rglob('*.png')}
        forms = forms_ad | forms_eh | forms_iz

        self.imgs = {}
        self.labels = {}
        self.authors = {}

        xml_files = list(IAM_XML_DIR_PATH.rglob('*.xml'))
        for xml_file in tqdm(xml_files):
            form_id, writer_id, lines = extract_lines_and_words_from_xml(xml_file.read_text())

            if writer_id not in self.shtg_authors:
                continue

            img = Image.open(forms[form_id])
            img = np.array(img)

            for line in lines:
                words = line['words']
                words_coords = [get_word_coords(word['element']) for word in words]

                words = [w for w, c in zip(words, words_coords) if c is not None]
                words_coords = [c for c in words_coords if c is not None]

                for start_idx, word in enumerate(words):
                    end_idx = start_idx
                    while end_idx < len(words) and sum(words_coords[start_idx:end_idx], words_coords[start_idx]).width_scaled(height) < min_width:
                        end_idx += 1
                        
                    coords = sum(words_coords[start_idx:end_idx], words_coords[start_idx])
                    if min_width < coords.width_scaled(height) < max_width and start_idx < end_idx:
                        img_crop = img[coords.y1:coords.y2, coords.x1:coords.x2]
                        img_crop = Image.fromarray(img_crop)
                        dst_path = self.output_dir / writer_id / f'{line["id"]}-{start_idx:02d}-{end_idx:02d}.png'

                        try:
                            text, dist = join_words([word['text'] for word in words[start_idx:end_idx]], line['text'])
                        except:
                            continue

                        if dist != 0:
                            # Somethimes there is a discrepancy between the collage of words and the 
                            # text contained in the text lines. In these cases we skip the sample
                            continue

                        self.imgs[dst_path.stem] = dst_path
                        self.labels[dst_path.stem] = text
                        self.authors[dst_path.stem] = writer_id

                        dst_path.parent.mkdir(exist_ok=True, parents=True)
                        img_crop.save(dst_path)

        self.data = self.generate_shtg_data()
        self.save_data_compressed()
        self.save_transcriptions(self.output_dir)

    def download_forms(self):
        if not IAM_FORMS_AD_DIR_PATH.exists():
            download_file(IAM_FORMS_AD_URL, IAM_FORMS_AD_TGZ_PATH)
            extract_tgz(IAM_FORMS_AD_TGZ_PATH, IAM_FORMS_AD_DIR_PATH, delete=True)

        if not IAM_FORMS_EH_DIR_PATH.exists():
            download_file(IAM_FORMS_EH_URL, IAM_FORMS_EH_TGZ_PATH)
            extract_tgz(IAM_FORMS_EH_TGZ_PATH, IAM_FORMS_EH_DIR_PATH, delete=True)

        if not IAM_FORMS_IZ_DIR_PATH.exists():
            download_file(IAM_FORMS_IZ_URL, IAM_FORMS_IZ_TGZ_PATH)
            extract_tgz(IAM_FORMS_IZ_TGZ_PATH, IAM_FORMS_IZ_DIR_PATH, delete=True)
    
    def delete(self):
        shutil.rmtree(self.output_dir)
        self.shtg_path.unlink()
