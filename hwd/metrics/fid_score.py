from .base_score import BaseScore, ProcessedDataset
import torch
from datasets.transforms import fid_our_transforms
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader

from .fid.inception import InceptionV3
from .fid.fid_score_crop64x64 import calculate_frechet_distance, calculate_fid_given_paths


class FIDScore(BaseScore):
    def __init__(self, dims=2048, device='cpu'):
        self.dims = dims
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]

        model = InceptionV3([block_idx])
        self.model = model.to(self.device)
        self.transform = fid_our_transforms

    def __call__(self, dataset1, dataset2, batch_size=128, verbose=False, ganwriting_script=False):
        if ganwriting_script:
            return self._compute_ganwriting_original_script(batch_size, verbose)
        return super().__call__(dataset1, dataset2, batch_size=128, verbose=verbose)

    def digest(self, dataset, batch_size=128, verbose=False):
        dataset.transform = self.transform
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ids, labels, features = FIDScore.get_activations(loader, self.model, self.device, verbose)
        return ProcessedDataset(ids, labels, features)

    def distance(self, data1, data2, **kwargs):
        mu1 = torch.mean(data1.features.T, dim=1).cpu().numpy()
        sigma1 = torch.cov(data1.features.T).cpu().numpy()
        mu2 = torch.mean(data2.features.T, dim=1).cpu().numpy()
        sigma2 = torch.cov(data2.features.T).cpu().numpy()
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def _compute_ganwriting_original_script(self, batch_size=128, verbose=False):
        cuda = self.device == 'cuda'

        print('WARNING: The "compute_ganwriting" method don\'t allow any kind of filter or transformations.')
        if verbose: print('Computing FID for dataset1 and dataset2')
        path1, path2 = str(self.dataset1.path), str(self.dataset2.path)
        fid_value = calculate_fid_given_paths((path1, path2), batch_size, cuda, self.dims)
        return fid_value

    # @staticmethod
    # def compute_statistics_of_dataset(loader, model, device, verbose=False):
    #     act = FIDScore.get_activations(loader, model, device, verbose)
    #     mu = torch.mean(act, dim=1)
    #     sigma = torch.cov(act)
    #
    #     return mu, sigma

    @staticmethod
    @torch.inference_mode()
    def get_activations(loader, model, device, verbose=False):
        model.eval()

        features = []
        labels = []
        for i, (images, authors) in enumerate(loader):
            images = images.to(device)

            pred = model(images)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            features.append(pred.squeeze(-1, -2).cpu())
            labels.append(list(authors))
            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)

        if verbose:
            print(' OK')

        features = torch.cat(features, dim=0)
        labels = sum(labels, [])
        ids = torch.arange(len(labels))
        return ids, labels, features

