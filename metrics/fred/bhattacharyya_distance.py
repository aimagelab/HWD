import numpy as np


def bhattacharyya_gaussian_distance(mean1, cov1, mean2, cov2):
    mean1 = mean1.unsqueeze(0).numpy()
    mean2 = mean2.unsqueeze(0).numpy()
    cov1 = cov1.numpy()
    cov2 = cov2.numpy()

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        np.sqrt((mean1 - mean2) @ np.linalg.inv(cov) @ (mean1 - mean2).T)[0][0]
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2
