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
This section describes how to evaluate the quality of styled handwritten text generation using various scores.

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

Each author's folder should contain `.png` images of their handwriting. To evaluate the Character Error Rate (CER), you must include a `transcriptions.json` file in the dataset. This file should be a dictionary where:

 - The **key** is the relative path to the image.
 - The **value** is the ground-truth text contained in the image.

Example structure of `transcriptions.json`:

```json
{
  "author1/sample1.png": "Hello world",
  "author1/sample2.png": "Handwritten text generation",
  "author2/sample1.png": "British Machine Vision Conference",
  "author2/sample2.png": "Artificial intelligence"
}
```
The `transcriptions.json` file is only required for CER evaluation and will be automatically parsed by the `FolderDataset` class if present in the dataset directory. Ensure all images referenced in the `transcriptions.json` file are in the corresponding folders.

Once your dataset is prepared, you can use the `FolderDataset` class to load images for evaluation:

```python
from hwd.datasets import FolderDataset

fakes = FolderDataset('/path/to/images/fake')
reals = FolderDataset('/path/to/images/real')
```

### Unfolded Dataset
Some evaluation metrics depend on whether the dataset is folded or unfolded. The unfold operation divides images into square segments, preserving the original height.

For an image of height $h$ and width $w$, the unfold operation splits the image into $n=⌊w/h⌋$ square images of size $h \times h$.

![unfold_5](https://github.com/user-attachments/assets/f49da3d9-692c-45cd-be86-c05928410a20)

```python
fakes = fakes.unfold()
reals = reals.unfold()
```
For FID and KID, images are cropped by default, as described in the paper. If you wish to evaluate using the entire line instead of cropping, you can unfold the dataset using the above method.

## HWD (Handwriting Distance)
The HWD is the primary evaluation score introduced in the paper. It compares two datasets (reference and generated) by resizing images to a height of 32 pixels and using the Euclidean distance between their features.

![HWD](https://github.com/user-attachments/assets/c64152c6-3414-4cb1-b4ab-a31202fe8fb4)

```python
from hwd.scores import HWDScore

hwd = HWDScore(height=32)
score = hwd(fakes, reals)
print(f"HWD Score: {score}")
```

## FID (Fréchet Inception Distance)
The FID compares the distributions of two datasets in the feature space of an InceptionNet pretrained on ImageNet. By default, images are cropped before evaluation.

![FID](https://github.com/user-attachments/assets/bd4e4538-0508-4f52-835d-4371c5e968ac)

```python
from hwd.scores import FIDScore

fid = FIDScore(height=32)
score = fid(fakes, reals)
print(f"FID Score: {score}")
```

## BFID (Binarized FID)
The BFID is a variant of the FID that operates on binarized images. This score is computed by applying Otsu's thresholding before evaluation.

```python
from hwd.scores import BFIDScore

bfid = BFIDScore(height=32)
score = bfid(fakes, reals)
print(f"BFID Score: {score}")
```

## KID (Kernel Inception Distance)
The KID measures differences between sets of images by using the maximum mean discrepancy (MMD). By default, images are cropped before evaluation.

```python
from hwd.scores import KIDScore

kid = KIDScore(height=32)
score = kid(fakes, reals)
print(f"KID Score: {score}")
```

## BKID (Binarized KID)
The BKID is a variant of the KID that operates on binarized images. This score is computed by applying Otsu's thresholding before evaluation.

```python
from hwd.scores import BKIDScore

bkid = BKIDScore(height=32)
score = bkid(fakes, reals)
print(f"BKID Score: {score}")
```

## CER (Character Error Rate)
The CER evaluates the character-level accuracy of generated handwritten text images by comparing their contained text against the ground-truth transcriptions. By default, the model `Microsoft/trocar-base-handwritten` is used.

```python
from hwd.scores import CERScore

# Load datasets
fakes = FolderDataset('/path/to/images/fake')  # Ensure this folder contains transcriptions.json

# Initialize CER score
cer = CERScore(height=64)

# Compute CER
score = cer(fakes)
print(f"CER Score: {score}")
```

## LPIPS (Learned Perceptual Image Patch Similarity)
The LPIPS measures perceptual differences between images by using feature activations from a deep network. The LPIPS score in this repo uses a custom implementation with the same backbone as HWD.

```python
from hwd.scores import LPIPSScore

lpips = LPIPSScore(height=32)
score = lpips(fakes, reals)
print(f"LPIPS Score: {score}")
```

## I-LPIPS (Intra-LPIPS)
The I-LPIPS evaluates the intra-image consistency by comparing style coherence between crops within the sample. This is also a custom implementation using the same backbone as HWD.

```python
from hwd.scores import IntraLPIPSScore

ilpips = IntraLPIPSScore(height=32)
score = ilpips(fakes)
print(f"I-LPIPS Score: {score}")
```

## GeometryScore
Python implementation of the algorithms from [the paper](https://arxiv.org/abs/1802.02664). If you use this algorithm in your research we kindly ask you to cite our work

```
@article{khrulkov2018geometry,
  title={Geometry {S}core: {A} {M}ethod {F}or {C}omparing {G}enerative {A}dversarial {N}etworks},
  author={Khrulkov, Valentin and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1802.02664},
  year={2018}
}
```

```python
from hwd.scores import GeometryScore

gs = GeometryScore(height=32, max_workers=8, n=1000)
score = gs(fakes, reals)
print(f"Geometry Score: {score}")
```