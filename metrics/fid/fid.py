import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader

from .inception import InceptionV3
from .fid_score_crop64x64 import calculate_frechet_distance, calculate_fid_given_paths


def compute_statistics_of_dataset(loader, model, device, verbose=False):
    act = get_activations(loader, model, device, verbose)
    mu = torch.mean(act, dim=1)
    sigma = torch.cov(act)

    return mu, sigma

@torch.no_grad()
def get_activations(loader, model, device, verbose=False):
    model.eval()

    pred_arr = []
    for i, (images, _) in enumerate(loader):
        images = images.to(device)

        pred = model(images)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr.append(pred.squeeze(-1, -2).cpu())
        if verbose:
            print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)

    if verbose:
        print(' OK')

    return torch.cat(pred_arr, dim=0).permute(1, 0)


class FID:
    def __init__(self, dataset1, dataset2, dims=2048, cuda=True):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dims = dims
        self.device = 'cuda' if cuda else 'cpu'

    def compute(self, batch_size=128, verbose=False, ganwriting_script=False):
        if ganwriting_script:
            return self._compute_ganwriting_original_script(batch_size, verbose)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]

        model = InceptionV3([block_idx])
        model = model.to(self.device)

        loader1 = DataLoader(self.dataset1, batch_size=batch_size, shuffle=False)
        loader2 = DataLoader(self.dataset2, batch_size=batch_size, shuffle=False)

        m1, s1 = compute_statistics_of_dataset(loader1, model, self.device, verbose)
        m2, s2 = compute_statistics_of_dataset(loader2, model, self.device, verbose)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value

    def _compute_ganwriting_original_script(self, batch_size=128, verbose=False):
        cuda = self.device == 'cuda'

        print('WARNING: The "compute_ganwriting" method don\'t allow any kind of filter or transformations.')
        if verbose: print('Computing FID for dataset1 and dataset2')
        path1, path2 = str(self.dataset1.path), str(self.dataset2.path)
        fid_value = calculate_fid_given_paths((path1, path2), batch_size, cuda, self.dims)
        return fid_value

