from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import json
import warnings
import math
import numpy as np
from tqdm import tqdm

class BaseDataset(Dataset):
    def __init__(self, path, imgs, authors, author_ids, nameset=None, transform=None, preprocess=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        self.path = Path(path)
        self.imgs = imgs
        self.authors = authors
        self.author_ids = author_ids
        self.nameset = nameset

        self.transform = transform
        self.preprocess = preprocess
        self.is_sorted = False

        self.labels = [None] * len(self.imgs)
        self.has_labels = False

        transcriptions_path = self.path / 'transcriptions.json'
        if transcriptions_path.exists():
            try:
                labels = json.loads(transcriptions_path.read_text())
                self.labels = [labels[str(img_path.relative_to(self.path))] for img_path in self.imgs]
                self.has_labels = True
            except KeyError:
                warnings.warn('Found the transcriptions.json with a wrong format. Please check the docs to know how to format the file')

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB') if isinstance(img_path, Path) else img_path 
        label = self.labels[index]
        author = self.authors[index]
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, author, label

    def __len__(self):
        return len(self.imgs)

    def sort(self, verbose=False):
        if self.is_sorted:
            return
        imgs_width = []
        for i, (img, label) in enumerate(self):
            imgs_width.append(img.size(2))
            if verbose:
                print(f'\rSorting {i + 1}/{len(self)} ', end='', flush=True)
        self.imgs = [x for _, x in sorted(zip(imgs_width, self.imgs), key=lambda pair: pair[0])]
        if verbose:
            print(' OK')
        self.is_sorted = True


    def _unfold_img(self, img, patch_width=None, stride=None, pad_img=True):
        image_array = np.array(img)
        
        # Get the dimensions of the image
        height, width, channels = image_array.shape
        patch_width = patch_width if patch_width is not None else height
        stride = stride if stride is not None else height

        if pad_img:
            new_width = math.ceil(width / stride) * stride
            remaining_width = new_width - width
            image_array = np.pad(
                image_array, 
                ((0, 0), (0, remaining_width), (0, 0)), 
                mode='constant', 
                constant_values=255
            )
            width = new_width
        
        # Initialize an empty list to store patches
        patches = []

        # Loop through the image width and extract patches
        for x_start in range(0, width - patch_width + 1, stride):
            # Set y_start to 0 and y_end to patch_height to cover the full height slice
            y_start = 0
            y_end = height
            
            # Extract the patch
            patch = image_array[y_start:y_end, x_start:x_start + patch_width, :]
            patches.append(patch)
        
        patches = [Image.fromarray(patch) for patch in patches]
        return patches

    def unfold(self, patch_width=None, stride=None, pad_img=False, verbose=False):
        imgs = []
        authors = []
        self.imgs_ids = []
        for i, (img, author) in tqdm(enumerate(zip(self.imgs, self.authors)), total=len(self.imgs), disable=not verbose):
            img = Image.open(img).convert('RGB')
            img_patches = self._unfold_img(img, patch_width=patch_width, stride=stride, pad_img=pad_img)
            imgs.extend(img_patches)
            authors.extend([author] * len(img_patches))
            self.imgs_ids.extend([i] * len(img_patches))
        self.imgs = imgs
        self.authors = authors
        self.labels = [None] * len(self.imgs)
        return self
