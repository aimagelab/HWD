import torch
from inception import InceptionV3
from fid_score_crop64x64 import calculate_frechet_distance
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d

def compute_statistics_of_dataset(loader, model, verbose=False):
    act = get_activations(loader, model, verbose)
    mu = torch.mean(act, dim=0)
    sigma = torch.cov(act, rowvar=False)

    return mu, sigma

def get_activations(loader, model, verbose=False):
    model.eval()

    pred_arr = []
    for i, (images, _) in enumerate(loader):
        pred = model(images)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr.append(pred.cpu().data.numpy().reshape(pred.size(0), -1))
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, len(loader)), end='', flush=True)

    if verbose:
        print(' done')

    return torch.stack(pred_arr)

class FID:
    def __init__(self, dataset1, dataset2, dims=2048, verbose=True, device=torch.cuda.is_available()):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dims = dims
        self.device = device
        self.verbose = verbose

    def compute(self, batch_size=128):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]

        model = InceptionV3([block_idx])
        model = model.to(self.device)

        loader1 = DataLoader(self.dataset1, batch_size=batch_size, shuffle=False)
        loader2 = DataLoader(self.dataset2, batch_size=batch_size, shuffle=False)

        m1, s1 = compute_statistics_of_dataset(loader1, model, self.verbose)
        m2, s2 = compute_statistics_of_dataset(loader2, model, self.verbose)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value
