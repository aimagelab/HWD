import unicodedata
import requests
import tarfile
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import functools
import shutil
import zipfile
import gzip
import json
import re

def download_file(url, filename, exist_ok=False):
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and exist_ok:
        return

    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    desc = f'Downloading {path.name}'
    if file_size == 0:
        desc += ' (Unknown total file size)'
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path


def extract_tgz(file_path, extract_path='.', delete=False):
    file_path = Path(file_path)
    # Extract the .tgz file to the specified directory
    print(f'Extracting {file_path.name} ... ', end='', flush=True)
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print('OK')
    if delete:
        file_path.unlink()

def extract_zip(file_path, extract_path='.', delete=False):
    # Extract .zip file to the specified directory
    print(f'Extracting {file_path.name} ... ', end='', flush=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_path)
    print('OK')
    if delete:
        file_path.unlink()

def simplify_text(text, charset):
    simplified_text = []
    for char in text:
        norm_char = unicodedata.normalize('NFD', char)
        if char in charset:
            simplified_text.append(char)
        else:
            for nchar in norm_char:
                if nchar in charset:
                    simplified_text.append(nchar)
    simplified_text = ''.join(simplified_text)
    
    return simplified_text

class BaseSHTGDataset():
    def __init__(self, load_style_samples, load_gen_sample, num_style_samples):
        self.load_style_samples = load_style_samples
        self.load_gen_sample = load_gen_sample
        self.num_style_samples = num_style_samples
        self.simplify_text = False

    def save_data_compressed(self, path=None):
        path = Path(path) if path is not None else self.shtg_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(self.data, f)

    def save_reference(self, path):
        path = Path(path)
        for sample in tqdm(self.data):
            img = Image.open(self.imgs[sample['gen_id']])
            dst = path / sample['dst']
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst)
        self.save_transcriptions(path)

    def save_transcriptions(self, path):
        path = Path(path)
        transcriptions = {}
        for sample in self.data:
            transcriptions[sample['dst']] = sample['text']
        with open(path / 'transcriptions.json', 'w') as f:
            json.dump(transcriptions, f)

    def check_compliance(self, path):
        path = Path(path)
        for sample in self.data:
            dst = path / sample['dst']
            if not dst.exists():
                return False
        return True

    def set_charset(self, charset):
        self.simplify_text = True
        self.charset = set(charset)
        return self
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        output = {}
        if self.simplify_text:
            output['gen_text'] = simplify_text(sample['text'], self.charset)
        else:
            output['gen_text'] = sample['text']
        output['author'] = Path(sample['dst']).parent.name
        output['dst_path'] = sample['dst']
        output['gen_id'] = sample['gen_id']
        output['style_ids'] = [sample['style_ids'][i % len(sample['style_ids'])] for i in range(self.num_style_samples)]
        output['style_imgs_path'] = [self.imgs[id] for id in output['style_ids']]
        output['style_imgs_text'] = [self.labels[id] for id in output['style_ids']]
        if self.load_style_samples:
            output['style_imgs'] = [Image.open(self.imgs[id]) for id in output['style_ids']]
        if self.load_gen_sample:
            output['gen_img'] = Image.open(self.imgs[sample['gen_id']])
        return output