from metrics import *
from datasets import *
from datasets.transforms import hwd_transforms, fid_our_transforms, gs_transforms
import random


# fake_path = '/home/vpippi/VATr/saved_images/vatr_same_words/Fake_test'  # 25766
# real_path = '/home/vpippi/VATr/saved_images/vatr_same_words/Real_test'  # 25766
# fake_path = '/home/vpippi/VATr/saved_images/vatr_rand_words/Fake_test'  # 25766
# real_path = '/home/vpippi/VATr/saved_images/vatr_rand_words/Real_test'  # 25766
# fake_path = '/home/vpippi/HiGAN+/eval_same_words/real'
# real_path = '/home/vpippi/HiGAN+/eval_same_words/fake'
# fake_path = '/home/vpippi/HiGAN+/eval_random_words/real'
# real_path = '/home/vpippi/HiGAN+/eval_random_words/fake'
fake_path = '/home/vpippi/Handwriting-Transformers/saved_images/Fake_test'
real_path = '/home/vpippi/Handwriting-Transformers/saved_images/Real_test'


def compute(src1, src2, score_class, transforms, max_len=25766):
    fake_dataset = FolderDataset(src1, transform=transforms)
    real_dataset = FolderDataset(src2, transform=transforms)
    fake_dataset.imgs = fake_dataset.imgs[:max_len]
    real_dataset.imgs = real_dataset.imgs[:max_len]
    fake_dataset.labels = fake_dataset.labels[:max_len]
    real_dataset.labels = real_dataset.labels[:max_len]
    score = score_class()
    res = score(fake_dataset, real_dataset, verbose=True)
    print(score_class.__name__, res)
    return res


print(fake_path)
print(real_path)
geometric_score = compute(fake_path, real_path, GeometricScore, gs_transforms)
vont_score = compute(fake_path, real_path, VontScore, hwd_transforms)
fid_score = compute(fake_path, real_path, FIDScore, fid_our_transforms)
kid_score = compute(fake_path, real_path, KIDScore, fid_our_transforms)
fid_inf_score = compute(fake_path, real_path, FIDInfScore, fid_our_transforms)

with open('results.txt', 'a') as f:
    f.write(f'{fake_path} {real_path}\n')
    f.write(f'VontScore {vont_score}\n')
    f.write(f'FIDScore {fid_score}\n')
    f.write(f'KIDScore {kid_score}\n')
    f.write(f'FIDInfScore {fid_inf_score}\n')
    f.write(f'GeometricScore {geometric_score}\n')
    f.write('\n')

print('Done')
