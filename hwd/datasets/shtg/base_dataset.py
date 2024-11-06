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

    desc = "(Unknown total file size)" if file_size == 0 else ""
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
    def __init__(self, load_style_samples, num_style_samples, scenario=None):
        self.load_style_samples = load_style_samples
        self.num_style_samples = num_style_samples
        self.scenario = scenario
        self.simplify_text = False

    @property
    def _data(self):
        if self.scenario is None:
            return self.data
        else:
            return [s for s in self.data if s['dst'].startswith(self.scenario)]

    def save_data_compressed(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(self.data, f)

    def _fill_data(self):
        from collections import defaultdict
        labels_mapping = defaultdict(list)
        for key, value in self.labels.items():
            labels_mapping[value].append(key)
        
        for sample in self._data:
            ids = labels_mapping[sample['word']]
            if len(ids) == 1:
                sample['gen_id'] = ids[0]
            elif len(ids) > 1:
                if sum([i in sample['dst'] for i in ids]) == 1:
                    for i in ids:
                        if i in sample['dst']:
                            sample['gen_id'] = i
                            break
                elif sum([Path(sample['dst']).parent.name in i for i in ids]) == 1:
                    print()
            else:
                raise ValueError

    def save_reference(self, path):
        for sample in self._data:
            print()

    def set_charset(self, charset):
        self.simplify_text = True
        self.charset = set(charset)
        return self
            
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data[idx]
        output = {}
        if self.simplify_text:
            output['gen_text'] = simplify_text(sample['word'], self.charset)
        else:
            output['gen_text'] = sample['word']
        output['author'] = Path(sample['dst']).parent.name
        output['dst_path'] = sample['dst']
        output['style_ids'] = [sample['style_ids'][i % len(sample['style_ids'])] for i in range(self.num_style_samples)]
        output['style_imgs_path'] = [self.imgs[id] for id in output['style_ids']]
        output['style_imgs_text'] = [self.labels[id] for id in output['style_ids']]
        if self.load_style_samples:
            output['style_imgs'] = [Image.open(self.imgs[id]) for id in output['style_ids']]
        return output