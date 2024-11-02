import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class SilhouetteScore():
    def distance(self, good, bad, metric='euclidean', **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        data = np.concatenate([good, bad]).reshape(-1, 1)
        clusters = np.concatenate([np.ones_like(good), np.zeros_like(bad)])
        return silhouette_score(data, clusters, metric=metric, **kwargs)


class CalinskiHarabaszScore():
    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        data = np.concatenate([good, bad]).reshape(-1, 1)
        clusters = np.concatenate([np.ones_like(good), np.zeros_like(bad)])
        return calinski_harabasz_score(data, clusters, **kwargs)


class DaviesBouldinScore():
    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        data = np.concatenate([good, bad]).reshape(-1, 1)
        clusters = np.concatenate([np.ones_like(good), np.zeros_like(bad)])
        return davies_bouldin_score(data, clusters, **kwargs)


class GrayZoneScore():
    def __init__(self, bins=40, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins

    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        bins = np.histogram(np.hstack((good, bad)), bins=self.bins)[1]
        good_bin_idx = np.digitize(good, bins)
        bad_bin_idx = np.digitize(bad, bins)
        good_bin_counts = np.bincount(good_bin_idx, minlength=len(bins))[:len(bins)]
        bad_bin_counts = np.bincount(bad_bin_idx, minlength=len(bins))[:len(bins)]
        overlap = np.sum(np.minimum(good_bin_counts, bad_bin_counts))
        return overlap / (len(good) + len(bad)) * 100.0


class EqualErrorRateScore():
    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        if good.max() < bad.min():
            return 0
        gray_zone_values = np.concatenate([good[good > bad.min()], bad[bad < good.max()]])
        miss = len(gray_zone_values)
        for val in gray_zone_values:
            good_miss = (good >= val).sum()
            bad_miss = (bad < val).sum()
            miss = min(miss, good_miss + bad_miss)
        return miss / (len(good) + len(bad)) / 2 * 100.0


class VITScore():
    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        lower_bound, upper_bound = np.percentile(np.concatenate([good, bad]), [25, 75])
        good = good.clip(lower_bound, upper_bound)
        bad = bad.clip(lower_bound, upper_bound)

        dist = np.abs(good[:, None] - bad[None, :]).sum()
        dist /= len(good) * len(bad)
        return dist / (upper_bound - lower_bound)
