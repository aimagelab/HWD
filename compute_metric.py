from metrics import FReD
from datasets import FakeDataset
from datasets.transforms import fid_ganwriting_tranforms, fid_our_tranforms, gs_tranforms, fred_tranforms
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import time

if __name__ == '__main__':
    path1 = '/home/vpippi/VATr/saved_images/vatr/Fake_test'
    path2 = '/home/vpippi/VATr/saved_images/vatr/Real_test'

    dataset1 = FakeDataset(path1, transform=fred_tranforms, max_samples=None)
    dataset2 = FakeDataset(path2, transform=fred_tranforms, max_samples=None)

    dataset1.sort()
    dataset2.sort()

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

    # gs = GeometricScore(dataset1, dataset2).compute(batch_size=256, verbose=True, parallel=True)
    # print('Geometric Score:', gs)  # 0.0003375288712080084

    for batch_size in [128, 64, 32, 16, 8, 4, 2, 1]:
        start = time.time()
        fred = FReD(dataset1, dataset2).compute(batch_size=batch_size, verbose=True)
        print(f'FReD ({batch_size}): {fred:.02f} time: {time.time() - start:.02f}')

