from metrics import FReDScore, InceptionScore, KIDScore, KReDScore
from datasets import FolderDataset
# from datasets.transforms import fid_ganwriting_tranforms, fid_our_tranforms, gs_tranforms, fred_tranforms
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from datasets.transforms import fred_transforms
import time
from datasets import CVLDataset

if __name__ == '__main__':
    path1 = '/home/vpippi/VATr/saved_images/vatr/Fake_train'
    path2 = '/home/vpippi/VATr/saved_images/vatr/Real_train'

    # dataset1 = FolderDataset(path1, transform=fred_tranforms, max_samples=None)
    # dataset2 = FolderDataset(path2, transform=fred_tranforms, max_samples=None)

    cvl_path = r'/home/shared/datasets/cvl-database-1-1'
    dataset1 = CVLDataset(cvl_path, transform=fred_transforms, max_samples=100)
    dataset2 = CVLDataset(cvl_path, transform=fred_transforms, max_samples=100)

    # fid_value = FID(dataset1, dataset2).compute(batch_size=256, verbose=True, ganwriting_script=True)
    # print('original GANwriting FID:', fid_value)
    # INTER_NEAREST:    14.847048027543224
    # INTER_LINEAR:
    # INTER_AREA:       14.850413989212797 - DEFAULT
    # INTER_CUBIC:      15.628173923192776
    # INTER_LANCZOS4:   14.879762926907773

    # dataset1 = FakeDataset(path1, transform=our_fid_tranforms)
    # dataset2 = FakeDataset(path2, transform=our_fid_tranforms)
    # fid_value = FID(dataset1, dataset2).compute(batch_size=256, verbose=True)
    # print('Our FID implementation:', fid_value)
    # NEAREST:  14.843643188476562 - DEFAULT
    # BOX:
    # BILINEAR: 16.6553955078125
    # HAMMING:
    # BICUBIC:  17.365203857421875
    # LANCZOS:  16.488265991210938

    # dataset1 = FakeDataset(path1, transform=ganwriting_fid_tranforms)
    # dataset2 = FakeDataset(path2, transform=ganwriting_fid_tranforms)
    # fid_value = FID(dataset1, dataset2).compute(batch_size=256, verbose=True)
    # print('GANwriting FID implementation:', fid_value)

    # gs_1 = GeometricScore(dataset1, dataset2).compute(batch_size=256, verbose=True, parallel=True)
    # print('Geometric Score:', gs_1)  # 0.0003375288712080084

    start = time.time()
    kred_score = KReDScore()
    a = kred_score(dataset1, dataset2)
    print(f'KReD: {a}')

    # start = time.time()
    # inception_score = InceptionScore()
    # a = inception_score(dataset1)
    # print(f'IS: {a}')
    #
    start = time.time()
    kid_score = KIDScore()
    a = kid_score(dataset1, dataset2)
    print(f'KID: {a}')

    # fred = FReDScore(dataset1, dataset2).compute(batch_size=128, verbose=True)
    # print(f'FReD (128): {fred:.02f} time: {time.time() - start:.02f}')

