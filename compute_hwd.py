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
    results = []
    assert fake_dataset.author_ids == real_dataset.author_ids
    fake_pd = score.digest(fake_dataset, verbose=True)
    real_pd = score.digest(real_dataset, verbose=True)
    for auth_id in fake_dataset.author_ids:
        res = score.distance(fake_pd[auth_id], real_pd[auth_id], verbose=True)
        print(score_class.__name__, auth_id, res)
        results.append(res)
    results = sum(results) / len(results)
    print(score_class.__name__, results)
    return results


print(fake_path)
print(real_path)
vont_score = compute(fake_path, real_path, VontScore, hwd_transforms, None)

print('Done')
