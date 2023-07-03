from metrics import HWDScore, FIDScore, KIDScore
from metrics import ProcessedDataset
from datasets import FolderDataset

real_dataset = FolderDataset(r'C:\Users\Vit\Desktop\Qualitativi VATR\168')
fake_dataset = FolderDataset(r'C:\Users\Vit\Desktop\Qualitativi VATR\169')

print('Real dataset:', len(real_dataset))
print('Fake dataset:', len(fake_dataset))

# score = HWDScore()
# real_dataset = score.digest(real_dataset)
# fake_dataset = score.digest(fake_dataset)
# result = score.distance(real_dataset, fake_dataset)
# print('HWD:', result)

# score = FIDScore()
# real_dataset = score.digest(real_dataset)
# fake_dataset = score.digest(fake_dataset)
# result = score.distance(real_dataset, fake_dataset)
# print('FID:', result)

score = KIDScore()
real_dataset = score.digest(real_dataset)
fake_dataset = score.digest(fake_dataset)
result = score.distance(real_dataset, fake_dataset)
print('KID:', result)
