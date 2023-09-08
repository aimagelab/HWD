# HWD: A Novel Evaluation Score for Styled Handwritten Text Generation

This repository contains the reference code and dataset for the paper [HWD: A Novel Evaluation Score for Styled Handwritten Text Generation](https://arxiv.org/abs/).
If you find it useful, please cite it as:
```
@inproceedings{pippi2023handwritten,
  title={{HWD: A Novel Evaluation Score for Styled Handwritten Text Generation}},
  author={Pippi, Vittorio and Quattrini, Fabio and and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle={Proceedings of the British Machine Vision Conference},
  year={2023}
}
```

## Installation
This is the list of python packages that we need to compute the score on windows with Python 3.9.13
```console
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip3 install opencv-python==4.7.0.72
pip3 install scipy==1.11.1
pip3 install botorch==0.8.5
pip3 install torchmetrics==0.11.4
pip3 install six==1.16.0
pip3 install gudhi==3.8.0
pip3 install matplotlib==3.7.1
pip3 list

```

## Example
Basic code to compute the score
```python
from metrics import HWDScore, FIDScore, KIDScore
from datasets import FolderDataset

real_dataset = FolderDataset('path/to/folder', extension='png')
fake_dataset = FolderDataset('path/to/folder', extension='png')

score = HWDScore()
result = score(real_dataset, fake_dataset)
print('HWD:', result)

score = FIDScore()
result = score(real_dataset, fake_dataset)
print('FID:', result)

score = KIDScore()
result = score(real_dataset, fake_dataset)
print('KID:', result)
```

To store intermediate results is it possible to use the `ProcessedDataset`:
```python
from metrics import HWDScore, ProcessedDataset
from datasets import FolderDataset

real_dataset = FolderDataset('path/to/folder', extension='png')
fake_dataset = FolderDataset('path/to/folder', extension='png')

score = HWDScore()
processed_real_dataset = score.digest(real_dataset)

# save on disk
processed_real_dataset.save('processed.pkl')
del processed_real_dataset

# load from disk
processed_real_dataset = ProcessedDataset.load('processed.pkl')
processed_fake_dataset = score.digest(fake_dataset)

result = score.distance(processed_real_dataset, processed_fake_dataset)
print('HWD:', result)
```
