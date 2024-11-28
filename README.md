# HWD: A Novel Evaluation Score for Styled Handwritten Text Generation


This repository contains the reference code and dataset for the paper [HWD: A Novel Evaluation Score for Styled Handwritten Text Generation](https://papers.bmvc2023.org/0007.pdf).
If you find it useful, please cite it as:
```
@inproceedings{pippi2023hwd,
  title={{HWD: A Novel Evaluation Score for Styled Handwritten Text Generation}},
  author={Pippi, Vittorio and Quattrini, Fabio and and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle={Proceedings of the British Machine Vision Conference},
  year={2023}
}
```

## Installation
```console
git clone https://github.com/aimagelab/HWD.git
cd HWD
python setup.py sdist bdist_wheel
pip install .
```

# Generation
Detailed instructions for generating styled handwritten text will be added in a future update.

# Evaluation
This section describes how to evaluate the quality of styled handwritten text generation using various metrics.

## Dataset
Organize your data in the following folder structure:

```
dataset/
├── author1/
│   ├── sample1.png
│   ├── sample2.png
│   └── ...
├── author2/
│   ├── sample1.png
│   ├── sample2.png
│   └── ...
└── ...
```

Each author's folder should contain `.png` images of their handwriting. Additionally, if you intend to evaluate Character Error Rate (CER), you must include a `transcriptions.json` file in the dataset. This file should be a dictionary where:

 - The key is the relative path to the image.
 - The value is the ground-truth text contained in the image.

Example structure of `transcriptions.json`:

```json
{
  "author1/sample1.png": "Hello world",
  "author1/sample2.png": "Handwritten text generation",
  "author2/sample1.png": "British Machine Vision Conference",
  "author2/sample2.png": "Artificial intelligence"
}
```
The `transcriptions.json` file is only required for CER evaluation and will be automatically parsed by the `FolderDataset` class if present in the dataset directory. Ensure that all images referenced in the `transcriptions.json` file are present in the corresponding folders.

Once your dataset is prepared, you can use the `FolderDataset` class to load images for evaluation:

```python
from hwd.datasets import FolderDataset

fakes = FolderDataset('/path/to/images/fake')
reals = FolderDataset('/path/to/images/real')
```

### Unfolded Dataset
Some evaluation metrics depend on whether the dataset is folded or unfolded. The unfold operation divides images into square segments, preserving the original height.

For an image of height $h$ and width $w$, the unfold operation splits the image into $n=⌊w/h⌋$ square images of size $h \times h$.

```python
fakes = fakes.unfold()
reals = reals.unfold()
```
For FID and KID, images are cropped as described in the paper by default. If you wish to evaluate using the entire line instead of cropping, you can unfold the dataset using the above method.

## HWD (Handwriting Distance)
The primary evaluation metric introduced in the paper. It compares two datasets (reference and generated) by resizing images to a height of 32 pixels and using a Euclidean distance metric.

```python
from hwd.scores import HWDScore

hwd = HWDScore(height=32)
score = hwd(fakes, reals)
print(f"HWD Score: {score}")
```

## FID (Frechet Inception Distance)
The FID compares the distributions of two datasets in the feature space of a pretrained model. By default, images are cropped before evaluation.

```python
from hwd.scores import FIDScore

fid = FIDScore(height=32)
score = fid(fakes, reals)
print(f"FID Score: {score}")
```

## BFID (Binarized FID)
A variant of FID that operates on binarized images for improved comparison of certain datasets. The binarized scores are computed by applying Otsu's thresholding before evaluation.

```python
from hwd.scores import BFIDScore

bfid = BFIDScore(height=32)
score = bfid(fakes, reals)
print(f"BFID Score: {score}")
```

## KID (Kernel Inception Distance)
The KID measures differences using maximum mean discrepancy (MMD). By default, images are cropped before evaluation.

```python
from hwd.scores import KIDScore

kid = KIDScore(height=32)
score = kid(fakes, reals)
print(f"KID Score: {score}")
```

## BKID (Binarized KID)
Binarized version of KID, emphasizing fine-grained details in the handwriting. The binarized scores are computed by applying Otsu's thresholding before evaluation.

```python
from hwd.scores import BKIDScore

bkid = BKIDScore(height=32)
score = bkid(fakes, reals)
print(f"BKID Score: {score}")
```

## CER (Character Error Rate)
The CERScore evaluates the character-level accuracy of generated handwriting by comparing the predicted text against ground-truth transcriptions. By default, the model `microsoft/trocr-base-handwritten` is used.

```python
from hwd.scores import CERScore

# Load datasets
fakes = FolderDataset('/path/to/images/fake')  # Ensure this folder contains transcriptions.json
reals = FolderDataset('/path/to/images/real')  # Ensure this folder contains transcriptions.json

# Initialize CER score
cer = CERScore(height=64)

# Compute CER
score = cer(fakes)
print(f"CER Score: {score}")
```

### LPIPS (Learned Perceptual Image Patch Similarity)
Measures perceptual differences between images using feature activations from a deep network. The LPIPS metric uses a custom implementation with the same backbone as HWD.

```python
from hwd.scores import LPIPSScore

lpips = LPIPSScore(height=32)
score = lpips(fakes, reals)
print(f"LPIPS Score: {score}")
```

## I-LPIPS (Intra-LPIPS)
Evaluates intra-author consistency by comparing style coherence within the same author’s samples. This is also a custom implementation using the same backbone as HWD.

```python
from hwd.scores import IntraLPIPSScore

ilpips = IntraLPIPSScore(height=32)
score = ilpips(fakes)
print(f"I-LPIPS Score: {score}")
```

