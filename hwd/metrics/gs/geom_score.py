from __future__ import absolute_import
from __future__ import print_function
from .utils import relative
from .utils import witness
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    for i in tqdm(range(n), disable=not verbose):
        rlts[i, :] = rlt(X, L_0, gamma, i_max)
    return rlts


def rlts_parallel(X, n=10, max_workers=4, verbose=False, **kwargs):
    """
    Parallel version of the 'rlts' function using concurrent.futures
    (ProcessPoolExecutor).
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(rlt, X, **kwargs) for _ in range(n)]

        for future in tqdm(as_completed(futures), total=n, disable=not verbose):
            res = future.result()
            results.append(res)

    rlts_array = np.vstack(results)
    return rlts_array


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
