import requests
import tarfile
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import functools
import shutil

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

class BaseSHTGDataset():
    def __init__(self, load_style_samples, num_style_samples):
        self.load_style_samples = load_style_samples
        self.num_style_samples = num_style_samples
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        output = {}
        output['gen_text'] = sample['word']
        output['author'] = Path(sample['dst']).parent.name
        output['dst_path'] = sample['dst']
        output['style_ids'] = sample['style_ids'][:self.num_style_samples]
        output['style_imgs_path'] = [self.imgs[id] for id in output['style_ids']]
        if self.load_style_samples:
            output['style_imgs'] = [Image.open(self.imgs[id]) for id in output['style_ids']]
        return output