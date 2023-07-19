from .base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import random
import xml.etree.ElementTree as ET


class IAMDataset(BaseDataset):
    def __init__(self, path, transform=None, nameset=None, dataset_type='words'):
        """
        Args:
            path (string): Path folder of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            author_ids (list, optional): List of authors to consider.
            nameset (string, optional): Name of the dataset.
            max_samples (int, optional): Maximum number of samples to consider.
        """
        super().__init__(path, transform, nameset)

        self.imgs = list(Path(path, dataset_type).rglob('*.png'))
        xml_files = list(Path(path, 'xmls').rglob('*.xml'))
        author_dict = {xml_file.stem: ET.parse(xml_file).getroot().attrib['writer-id'] for xml_file in xml_files}

        self.author_ids = list(set(author_dict.values()))
        self.labels = [author_dict[img.parent.stem] for img in self.imgs]


if __name__ == '__main__':
    # iam_path = r'/mnt/beegfs/work/FoMo_AIISDH/datasets/IAM'
    iam_path = r'/home/shared/datasets/IAM'

    dataset = IAMDataset(iam_path)
    print(len(dataset))
    print(dataset[0][1])
