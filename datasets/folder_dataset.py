from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import numpy as np
import math


class FolderDataset(BaseDataset):
    def __init__(self, path, extension='png', **kwargs):
        super(FolderDataset, self).__init__(path, **kwargs)
        self.path = Path(path)
        self.imgs = list(Path(path).rglob(f'*.{extension}'))
        assert len(self.imgs) > 0, 'No images found.'
        self.labels = [img.parent.name for img in self.imgs]
        self.author_ids = sorted(set(self.labels))

class UnfoldDataset(FolderDataset):
    def __init__(self, path, extension='png', patch_width=None, stride=None, pad_img=True, **kwargs):
        super().__init__(path, extension, **kwargs)

        tmp_imgs = []
        tmp_labels = []
        for img, label in zip(self.imgs, self.labels):
            img = Image.open(img).convert('RGB')
            img_patches = self._unfold_img(img, pad_img=pad_img)
            tmp_imgs.extend(img_patches)
            tmp_labels.extend([label] * len(img_patches))
        self.imgs = tmp_imgs
        self.labels = tmp_labels
    
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
