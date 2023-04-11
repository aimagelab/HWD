from .base_score import BaseScore
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class SilhouetteScore(BaseScore):
    def distance(self, good, bad, metric='euclidean', **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        data = np.concatenate([good, bad]).reshape(-1, 1)
        clusters = np.concatenate([np.ones_like(good), np.zeros_like(bad)])
        return silhouette_score(data, clusters, metric=metric, **kwargs)

class CalinskiHarabaszScore(BaseScore):
    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        data = np.concatenate([good, bad]).reshape(-1, 1)
        clusters = np.concatenate([np.ones_like(good), np.zeros_like(bad)])
        return calinski_harabasz_score(data, clusters, **kwargs)

class DaviesBouldinScore(BaseScore):
    def distance(self, good, bad, **kwargs):
        good = np.array(good)
        bad = np.array(bad)
        data = np.concatenate([good, bad]).reshape(-1, 1)
        clusters = np.concatenate([np.ones_like(good), np.zeros_like(bad)])
        return davies_bouldin_score(data, clusters, **kwargs)
