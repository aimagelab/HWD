# Installation
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

# Example
Basic code to compute the score
```python
from metrics import HWDScore, FIDScore
from datasets import FolderDataset

real_dataset = FolderDataset('path/to/folder', extension='png')
fake_dataset = FolderDataset('path/to/folder', extension='png')

score = HWDScore()
result = score(real_dataset, fake_dataset)
print('HWD:', result)

score = FIDScore()
result = score(real_dataset, fake_dataset)
print('FID:', result)
```

To store intermediate results is it possible to save them:
```python
from metrics import HWDScore, ProcessedDataset
from datasets import FolderDataset

real_dataset = FolderDataset('path/to/folder', extension='png')
fake_dataset = FolderDataset('path/to/folder', extension='png')

score = HWDScore()
preprocessed_real_dataset = score.digest(real_dataset)

# save on disk
preprocessed_real_dataset.save('preprocessed.pkl')
del preprocessed_real_dataset

# load from disk
preprocessed_real_dataset = ProcessedDataset.load('preprocessed.pkl')
preprocessed_fake_dataset = score.digest(fake_dataset)

result = score.distance(preprocessed_real_dataset, preprocessed_fake_dataset)
print('HWD:', result)
```