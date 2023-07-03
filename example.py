from metrics import HWD, VontScore
from datasets import FolderDataset

real_dataset = FolderDataset(r'C:\Users\Vit\Desktop\Qualitativi VATR\168')
fake_dataset = FolderDataset(r'C:\Users\Vit\Desktop\Qualitativi VATR\169')

score = HWD()
real_dataset = score.digest(real_dataset)
fake_dataset = score.digest(fake_dataset)
result = score.distance(real_dataset, fake_dataset)
print('HWD:', result)

score = VontScore()
real_dataset = score.digest(real_dataset)
fake_dataset = score.digest(fake_dataset)
result = score.distance(real_dataset, fake_dataset)
print('Vont:', result)
