"""Kiểm tra src/evaluation/analysis.py, đối chiếu sklearn / scipy / công thức trực tiếp.
Chạy: python tests/check_analysis.py
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import log_loss as sk_log_loss

from _helpers import main
from src.evaluation.analysis import (fisher_scores, log_loss, overconfident_error_rate,
                                     probabilistic_scores, spearman)


def check_probabilistic_metrics():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(10), 300)
    y = rng.integers(0, 10, 300)
    assert np.isclose(log_loss(y, P), sk_log_loss(y, P, labels=range(10)))
    P_sure = np.eye(10)[(y + 1) % 10] * 0.995 + 0.0005             # sai hết và tự tin 99,5%
    assert overconfident_error_rate(y, P_sure) == 1.0
    s = probabilistic_scores(y, P_sure)
    assert s["overconfident_share_of_errors"] == 1.0 and np.isclose(s["mean_confidence_errors"], 0.9955)
    s_ok = probabilistic_scores(y, np.eye(10)[y])
    assert s_ok["mean_confidence_errors"] is None and s_ok["overconfident_share_of_errors"] == 0.0


def check_spearman_skips_nan():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 200).astype(float)
    b = a + rng.normal(0, 2, 200)
    a[::7] = np.nan
    ok = np.isfinite(a)
    assert np.isclose(spearman(a, b), spearmanr(a[ok], b[ok]).statistic), "phải bỏ vị trí NaN"


def check_fisher_scores_against_direct_formula():
    rng = np.random.default_rng(0)
    X = np.zeros((500, 784))
    X[:, 300:500] = rng.random((500, 200)) * (rng.random((500, 200)) < 0.4)
    y = rng.integers(0, 10, 500)
    F = fisher_scores(X, y)
    p = 400
    mk = np.array([X[y == k, p].mean() for k in range(10)])
    vk = np.array([X[y == k, p].var() for k in range(10)])
    nk = np.bincount(y, minlength=10)
    expected = (nk * (mk - X[:, p].mean()) ** 2).sum() / (nk * vk).sum()
    assert np.isclose(F[p], expected)
    assert np.isnan(F[0]) and np.isnan(F).sum() == 584              # pixel không có mực


CHECKS = [
    check_probabilistic_metrics,
    check_spearman_skips_nan,
    check_fisher_scores_against_direct_formula,
]

if __name__ == "__main__":
    main(CHECKS, "analysis")
