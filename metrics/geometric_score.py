from .base_score import BaseScore
from . import gs
import numpy as np
from torch.utils.data import DataLoader


class GeometricScore(BaseScore):
    def __init__(self, dataset1, dataset2, n=100, L_0=32, i_max=10, gamma=1.0 / 8):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.n = n
        self.L_0 = L_0
        self.i_max = i_max
        self.gamma = gamma

    @staticmethod
    def _concatenate_loader(loader, verbose=False):
        tmp_images = []
        for i, (images, _) in enumerate(loader):
            tmp_images.append(images)
            if verbose:
                print(f'\rReading images {i + 1}/{len(loader)}', end='', flush=True)

        if verbose:
            print(' OK')
        return np.concatenate(tmp_images, axis=0)

    def digest(self, dataset, batch_size=128, verbose=False, parallel=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        data = self._concatenate_loader(loader, verbose)

        if parallel:
            raise NotImplementedError('Parallel computation is not working implemented yet')
            # return gs.rlts_parallel(data, n=self.n, L_0=self.L_0, i_max=self.i_max, gamma=self.gamma, verbose=verbose)
        return gs.rlts(data, n=self.n, L_0=self.L_0, i_max=self.i_max, gamma=self.gamma, verbose=verbose)

    def distance(self, data1, data2):
        raise gs.geom_score(data1, data2)
