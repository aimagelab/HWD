from .base_score import BaseScore, ProcessedDataset
from .fid_infinity.score_infinity import randn_sampler, load_inception_net, load_path_statistics, \
    compute_path_statistics, numpy_calculate_frechet_distance, LinearRegression
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FIDInfScore(BaseScore):
    def __init__(self, dims=2048, device='cuda'):
        self.dims = dims
        self.device = device
        self.model = load_inception_net()
        self.model.to(self.device)

    def digest(self, dataset, batch_size=128, verbose=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ids, labels, features = FIDInfScore.get_activations(loader, self.model, self.device, verbose)
        return ProcessedDataset(ids, labels, features)

    def distance(self, data1, data2, batch_size=50, min_fake=2, num_points=15, **kwargs):
        real_act = data1.features.cpu().numpy()
        fake_act = data2.features.cpu().numpy()
        real_m, real_s = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
        # fake_m, fake_s = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)

        num_fake = len(fake_act)
        assert num_fake > min_fake, \
            'number of fake data must be greater than the minimum point for extrapolation'

        fids = []

        # Choose the number of images to evaluate FID_N at regular intervals over N
        fid_batches = np.linspace(min_fake, num_fake, num_points).astype('int32')

        # Evaluate FID_N
        for fid_batch_size in fid_batches:
            # sample with replacement
            np.random.shuffle(fake_act)
            fid_activations = fake_act[:fid_batch_size]
            m, s = np.mean(fid_activations, axis=0), np.cov(fid_activations, rowvar=False)
            FID = numpy_calculate_frechet_distance(m, s, real_m, real_s)
            fids.append(FID)
        fids = np.array(fids).reshape(-1, 1)

        # Fit linear regression
        reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
        fid_infinity = reg.predict(np.array([[0]]))[0, 0]

        return fid_infinity

    @staticmethod
    @torch.no_grad()
    def get_activations(loader, model, device, verbose=False):
        pool = []
        # logits = []
        labels = []

        for i, (images, authors) in enumerate(loader):
            images = images.to(device)
            pool_val, logits_val = model(images)
            pool += [pool_val.cpu()]
            # logits += [F.softmax(logits_val, 1)]
            labels.append(list(authors))
            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)} ', end='', flush=True)

        if verbose:
            print('OK')

        features = torch.cat(pool, 0)
        # logits = torch.cat(logits, 0).cpu().numpy()
        labels = sum(labels, [])
        ids = torch.arange(len(labels))
        return ids, labels, features
