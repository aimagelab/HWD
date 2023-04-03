from metrics import FID, GeometricScore
from datasets import FakeDataset
from datasets.transforms import ResizeHeight, CropStartSquare, ResizeSquare, CropStart
from torchvision.transforms import Compose, ToTensor

if __name__ == '__main__':
    transform = Compose([
        # CropStart(64),
        CropStartSquare(),
        ResizeSquare(299),
        ToTensor()
    ])
    dataset1 = FakeDataset('/home/vpippi/VATr/saved_images/vatr/Fake_train', transform=transform, max_samples=10000)
    dataset2 = FakeDataset('/home/vpippi/VATr/saved_images/vatr/Real_train', transform=transform, max_samples=10000)
    fid = FID(dataset1, dataset2)
    fid_value = fid.compute_slow(batch_size=256, verbose=True)
    print(fid_value)

