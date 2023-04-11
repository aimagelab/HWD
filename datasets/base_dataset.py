from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import copy

class BaseDataset(Dataset):
    def __init__(self, path, transform=None, author_ids=None, nameset=None, max_samples=None):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        if nameset is not None:
            raise NotImplementedError('Nameset is not implemented yet.')

        self.path = path
        self.imgs = []
        self.labels = []
        self.transform = transform
        self.all_author_ids = author_ids
        self.author_ids = author_ids
        self.nameset = nameset
        self.max_samples = max_samples
        self.is_sorted = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the target class.
        """
        img = self.filtered_imgs[index]
        label = self.filtered_labels[index]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filtered_imgs)

    @property
    def filtered_labels(self):
        return [label for label in self.labels if label in self.author_ids]

    @property
    def filtered_imgs(self):
        return [img for img, label in zip(self.imgs, self.labels) if label in self.author_ids]


    def split(self, ratio=0.5):
        """
        Split the dataset in two halfs.
        """
        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        dataset1.imgs = dataset1.filtered_imgs
        dataset2.imgs = dataset2.filtered_imgs
        dataset1.labels = dataset1.filtered_labels
        dataset2.labels = dataset2.filtered_labels

        size = int(len(self) * ratio)
        mask = np.array([0] * size + [1] * (len(self) - size))
        np.random.shuffle(mask)
        dataset1.imgs = [img for img, idx in zip(dataset1.imgs, mask) if idx == 0]
        dataset2.imgs = [img for img, idx in zip(dataset2.imgs, mask) if idx == 1]
        dataset1.labels = [label for label, idx in zip(dataset1.labels, mask) if idx == 0]
        dataset2.labels = [label for label, idx in zip(dataset2.labels, mask) if idx == 1]
        return dataset1, dataset2

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
