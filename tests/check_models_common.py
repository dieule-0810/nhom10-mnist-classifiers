"""Kiểm tra mốc tâm gần nhất (NearestCentroidNP), đối chiếu sklearn.
Chạy: python tests/check_models_common.py
"""

import warnings

import numpy as np
from sklearn.neighbors import NearestCentroid

from _helpers import main
from src.models.nearest_centroid import NearestCentroidNP


def _toy():
    """4 lớp Gauss trong 6 chiều, chồng lấn một phần."""
    rng = np.random.default_rng(0)
    means = rng.normal(0, 1.5, (4, 6))
    X = np.vstack([rng.normal(means[k], 1.0, (150, 6)) for k in range(4)])
    return X, np.repeat(np.arange(4), 150)


def _fake_images(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 28, 28), dtype=np.float32)
    X[:, 4:24, 4:24] = (rng.random((n, 20, 20)) > 0.7) * rng.random((n, 20, 20))
    return X.reshape(n, -1), rng.integers(0, 10, n)


def check_nearest_centroid_matches_sklearn():
    X, y = _toy()
    assert np.array_equal(NearestCentroidNP().fit(X, y).predict(X), NearestCentroid().fit(X, y).predict(X))
    Xi, yi = _fake_images()
    with warnings.catch_warnings():                            # sklearn có thể cảnh báo khi có cột hằng
        warnings.simplefilter("ignore", UserWarning)
        ref = NearestCentroid().fit(Xi, yi).predict(Xi)
    assert np.array_equal(NearestCentroidNP().fit(Xi, yi).predict(Xi), ref)


CHECKS = [
    check_nearest_centroid_matches_sklearn,
]

if __name__ == "__main__":
    main(CHECKS, "models_common")
