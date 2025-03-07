# Evaluation

## Index
1. [Datasets](#datasets)
   - [GeneratedDataset](#generateddataset)
   - [FolderDataset](#folderdataset)
   - [Unfolded Dataset](#unfolded-dataset)
2. [Evaluation Metrics](#evaluation-metrics)
   - [HWD (Handwriting Distance)](#hwd-handwriting-distance)
   - [FID (Fréchet Inception Distance)](#fid-fréchet-inception-distance)
   - [BFID (Binarized FID)](#bfid-binarized-fid)
   - [KID (Kernel Inception Distance)](#kid-kernel-inception-distance)
   - [BKID (Binarized KID)](#bkid-binarized-kid)
   - [CER (Character Error Rate)](#cer-character-error-rate)
   - [LPIPS (Learned Perceptual Image Patch Similarity)](#lpips-learned-perceptual-image-patch-similarity)
   - [I-LPIPS (Intra-LPIPS)](#i-lpips-intra-lpips)
   - [GeometryScore](#geometryscore)

This section describes how to evaluate the quality of styled handwritten text generation using various scores.

## Datasets
### GeneratedDataset
The `GeneratedDataset` class provides an easy way to download and use the images generated during the publication of the [Emuru](https://huggingface.co/blowing-up-groundhogs/emuru) paper. To use this dataset, pass a string formatted as `{dataset}__{model}` to the `GeneratedDataset` class. The available datasets can be found at: [HWD Releases - Generated Dataset](https://github.com/aimagelab/HWD/releases/tag/generated).

Example usage:

```python
from hwd.datasets import GeneratedDataset

fakes = GeneratedDataset('iam_words__emuru')
reals = GeneratedDataset('iam_words__reference')
```

### FolderDataset
The `FolderDataset` class allows users to evaluate handwritten text images stored locally. To use this dataset format, you need to structure your dataset with each author's handwriting samples placed in separate folders. This setup is particularly useful for evaluating handwritten text generation models where individual writing styles must be maintained.

Your dataset should be organized as follows:

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

Each author's folder should contain `.png` images representing their handwriting. If you intend to evaluate the Character Error Rate (CER), you must include a `transcriptions.json` file in the dataset. This file should be a dictionary where:

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

For an image of height $h$ and width $w$, the unfold operation splits the image into $n=\lfloor w/h \rfloor$ square images of size $h \times h$.

![unfold_5](https://github.com/user-attachments/assets/f49da3d9-692c-45cd-be86-c05928410a20)

```python
fakes = fakes.unfold()
reals = reals.unfold()
```
For FID and KID, images are cropped by default, as described in the paper. If you wish to evaluate using the entire line instead of cropping, you can unfold the dataset using the above method.


## Evaluation Metrics
### HWD (Handwriting Distance)
The HWD is the primary evaluation score introduced in the paper. It compares two datasets (reference and generated) by resizing images to a height of 32 pixels and using the Euclidean distance between their features.

![HWD](https://github.com/user-attachments/assets/c64152c6-3414-4cb1-b4ab-a31202fe8fb4)

```python
from hwd.scores import HWDScore

hwd = HWDScore(height=32)
score = hwd(fakes, reals)
print(f"HWD Score: {score}")
```

### FID (Fréchet Inception Distance)
The FID compares the distributions of two datasets in the feature space of an InceptionNet pretrained on ImageNet. By default, images are cropped before evaluation.

![FID](https://github.com/user-attachments/assets/bd4e4538-0508-4f52-835d-4371c5e968ac)

```python
from hwd.scores import FIDScore

fid = FIDScore(height=32)
score = fid(fakes, reals)
print(f"FID Score: {score}")
```

### BFID (Binarized FID)
The BFID is a variant of the FID that operates on binarized images. This score is computed by applying Otsu's thresholding before evaluation.

```python
from hwd.scores import BFIDScore

bfid = BFIDScore(height=32)
score = bfid(fakes, reals)
print(f"BFID Score: {score}")
```

### KID (Kernel Inception Distance)
The KID measures differences between sets of images by using the maximum mean discrepancy (MMD). By default, images are cropped before evaluation.

```python
from hwd.scores import KIDScore

kid = KIDScore(height=32)
score = kid(fakes, reals)
print(f"KID Score: {score}")
```

### BKID (Binarized KID)
The BKID is a variant of the KID that operates on binarized images. This score is computed by applying Otsu's thresholding before evaluation.

```python
from hwd.scores import BKIDScore

bkid = BKIDScore(height=32)
score = bkid(fakes, reals)
print(f"BKID Score: {score}")
```

### CER (Character Error Rate)
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

### LPIPS (Learned Perceptual Image Patch Similarity)
The LPIPS measures perceptual differences between images by using feature activations from a deep network. The LPIPS score in this repo uses a custom implementation with the same backbone as HWD.

```python
from hwd.scores import LPIPSScore

lpips = LPIPSScore(height=32)
score = lpips(fakes, reals)
print(f"LPIPS Score: {score}")
```

### I-LPIPS (Intra-LPIPS)
The I-LPIPS evaluates the intra-image consistency by comparing style coherence between crops within the sample. This is also a custom implementation using the same backbone as HWD.

```python
from hwd.scores import IntraLPIPSScore

ilpips = IntraLPIPSScore(height=32)
score = ilpips(fakes)
print(f"I-LPIPS Score: {score}")
```

### GeometryScore
Python implementation of the algorithms from [the paper](https://arxiv.org/abs/1802.02664).

```python
from hwd.scores import GeometryScore

gs = GeometryScore(height=32, max_workers=8, n=1000)
score = gs(fakes, reals)
print(f"Geometry Score: {score}")
```

