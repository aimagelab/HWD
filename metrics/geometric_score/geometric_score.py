import numpy as np
from . import gs
from pathlib import Path
from torch.utils.data import DataLoader


class GeometricScore:
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

    def compute(self, batch_size=128, verbose=False, parallel=False):
        loader1 = DataLoader(self.dataset1, batch_size=batch_size, shuffle=False)
        loader2 = DataLoader(self.dataset2, batch_size=batch_size, shuffle=False)

        data1 = self._concatenate_loader(loader1, verbose)
        data2 = self._concatenate_loader(loader2, verbose)

        if parallel:
            raise NotImplementedError('Parallel computation is not working implemented yet')
            # rlt1 = gs.rlts_parallel(data1, n=self.n, L_0=self.L_0, i_max=self.i_max, gamma=self.gamma, verbose=verbose)
            # rlt2 = gs.rlts_parallel(data2, n=self.n, L_0=self.L_0, i_max=self.i_max, gamma=self.gamma, verbose=verbose)
        else:
            rlt1 = gs.rlts(data1, n=self.n, L_0=self.L_0, i_max=self.i_max, gamma=self.gamma, verbose=verbose)
            rlt2 = gs.rlts(data2, n=self.n, L_0=self.L_0, i_max=self.i_max, gamma=self.gamma, verbose=verbose)

        return gs.geom_score(rlt1, rlt2)
