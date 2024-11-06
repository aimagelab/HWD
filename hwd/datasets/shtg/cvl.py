from PIL import Image
from .base_dataset import BaseSHTGDataset, download_file, extract_zip
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
import json
import gzip

SHTG_CVL_LINES_URL = 'https://github.com/aimagelab/HWD/releases/download/cvl/shtg_cvl_lines.json.gz'
SHTG_CVL_WORDS_URL = 'https://github.com/aimagelab/HWD/releases/download/cvl/shtg_cvl_words.json.gz'
SHTG_CVL_LINES_PATH = Path('.cache/cvl/shtg_cvl_lines.json.gz')
SHTG_CVL_WORDS_PATH = Path('.cache/cvl/shtg_cvl_words.json.gz')

CVL_URL = 'https://zenodo.org/records/1492267/files/cvl-database-1-1.zip?download=1'
CVL_ZIP_PATH = Path('.cache/cvl/cvl-database-1-1.zip')
CVL_DIR_PATH = Path('.cache/cvl/cvl-database-1-1')


def extract_lines_from_xml(xml_string):
    words = extract_words_from_xml(xml_string)
    tmp_lines = defaultdict(list)
    for word in words:
        writer_n, page_n, line_n, word_n = word['id'].split('-')
        line_id = '-'.join((writer_n, page_n, line_n))
        tmp_lines[line_id].append((int(word_n), word['text']))
    
    lines = []
    for line_id, line in tmp_lines.items():
        line = sorted(line)
        text = ' '.join(word for _, word in line)
        lines.append({'id':line_id, 'text':text})
    return lines


def extract_words_from_xml(xml_string):
    root = ET.fromstring(xml_string)
    
    # Define namespace
    ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}
    
    words = []
    
    # Find all AttrRegion elements with attrType="1" and a "text" attribute
    for attr_region in root.findall('.//ns:AttrRegion[@attrType="1"]', ns):
        word_id = attr_region.get("id")
        word_text = attr_region.get("text")
        if word_text:
            words.append({"id": word_id, "text": word_text})
    
    return words


class CVL(BaseSHTGDataset):
    def __init__(self, extract_fn, shtg_url, shtg_path, load_style_samples=True, num_style_samples=1):
        super().__init__(load_style_samples=load_style_samples, num_style_samples=num_style_samples)

        if not CVL_DIR_PATH.exists():
            download_file(CVL_URL, CVL_ZIP_PATH)
            extract_zip(CVL_ZIP_PATH, CVL_DIR_PATH.parent, delete=True)

        self.imgs = {'-'.join(img_path.stem.split('-')[:4]): img_path for img_path in CVL_DIR_PATH.rglob('*.tif')}

        self.samples = []
        xmls_path = CVL_DIR_PATH / 'testset' / 'xml'
        for xml_file in xmls_path.rglob('*.xml'):
            self.samples.extend(extract_fn(xml_file.read_text(encoding="ISO-8859-1")))

        self.data = []
        self.shtg_path = shtg_path
        download_file(shtg_url, shtg_path, exist_ok=True)
        with gzip.open(shtg_path, 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        self.labels = {sample['id']: sample['text'] for sample in self.samples}


    def generate_shtg_data(self):
        data = []
        for sample_id, text in tqdm(self.labels.items()):
            writer_n = sample_id.split('-')[0]

            if len(text) < 3:
                continue

            style_ids = []
            for sample_id_tgt, text_tgt in self.labels.items():
                writer_n_tgt = sample_id_tgt.split('-')[0]
                if writer_n == writer_n_tgt and sample_id != sample_id_tgt and len(text_tgt) > 2:
                    style_ids.append(sample_id_tgt)

            data.append({
                'text': text,
                'gen_id': sample_id,
                'dst': f'{writer_n}/{sample_id}.png',
                'style_ids': style_ids,
            })
        return data


class CVLWords(CVL):
    def __init__(self, load_style_samples=True, num_style_samples=1):
        super().__init__(extract_words_from_xml, SHTG_CVL_WORDS_URL, SHTG_CVL_WORDS_PATH, 
                         load_style_samples, num_style_samples)


class CVLLines(CVL):
    def __init__(self, load_style_samples=True, num_style_samples=1):
        super().__init__(extract_lines_from_xml, SHTG_CVL_LINES_URL, SHTG_CVL_LINES_PATH, 
                         load_style_samples, num_style_samples)


class CVLLinesFromWords(CVL):
    def __init__(self, load_style_samples=True, num_style_samples=1):
        raise NotImplementedError