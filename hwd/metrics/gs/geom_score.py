from __future__ import absolute_import
from __future__ import print_function
from .utils import relative
from .utils import witness
import numpy as np


def rlt(X, L_0=64, gamma=None, i_max=100):
    """
      This function implements Algorithm 1 for one sample of landmarks.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing RLT(i, 1, X, L)
      for randomly sampled landmarks.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError('X should be a numpy array')
    if len(X.shape) != 2:
        raise ValueError('X should be 2d array, got shape {}'.format(X.shape))
    N = X.shape[0]
    if gamma is None:
        gamma = 1.0 / 128 * N / 5000
    I_1, alpha_max = witness(X, L_0=L_0, gamma=gamma)
    res = relative(I_1, alpha_max, i_max=i_max)
    return res


def rlts_parallel(X, L_0=64, gamma=None, i_max=100, n=1000, num_workers=8, n_multiplier=1, verbose=False):
    from concurrent.futures import ThreadPoolExecutor

    def print_rlt(i):
        inner_res = []
        for _ in range(n_multiplier):
            inner_res.append(rlt(X, L_0, gamma, i_max))
            counter += 1
            if verbose:
                print(f'Computing RLT {i}/{n}')
        return inner_res

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(print_rlt, [i for i in range(n)])
    results = sum(results, [])
    print(' OK')
    return np.stack(results, axis=0)


def rlts(X, L_0=64, gamma=None, i_max=100, n=1000, verbose=False):
    """
      This function implements Algorithm 1.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.
      n: int, number of samples
    Returns
      An array of size (n, i_max) containing RLT(i, 1, X, L)
      for n collections of randomly sampled landmarks.
    """
    rlts = np.zeros((n, i_max))
    for i in range(n):
        if verbose:
            print(f'\rComputing RLT {i+1}/{n}', end='', flush=True)
        rlts[i, :] = rlt(X, L_0, gamma, i_max)

    if verbose:
        print(' OK')
    return rlts


def geom_score(rlts1, rlts2):
    """
      This function implements Algorithm 2.

    Args:
       rlts1 and rlts2: arrays as returned by the function "rlts".
    Returns
       Float, a number representing topological similarity of two datasets.

    """
    mrlt1 = np.mean(rlts1, axis=0)
    mrlt2 = np.mean(rlts2, axis=0)
    return np.sum((mrlt1 - mrlt2) ** 2)
