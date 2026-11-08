"""Kiểm tra Logistic Regression, đối chiếu phần tự cài với lời giải độc lập (sklearn, sai phân hữu hạn).
Chạy: python tests/check_logistic_regression.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from _helpers import main
from src.models._numeric import add_bias, one_hot
from src.models.logistic_regression import SoftmaxRegressionNP


def _toy():
    """4 lớp Gauss trong 6 chiều, chồng lấn một phần."""
    rng = np.random.default_rng(0)
    K, n, d = 4, 150, 6
    means = rng.normal(0, 1.5, (K, d))
    X = np.vstack([rng.normal(means[k], 1.0, (n, d)) for k in range(K)])
    y = np.repeat(np.arange(K), n)
    return X, y


def check_gradient_matches_finite_difference():
    """∇J giải tích so với sai phân trung tâm, ở float64."""
    X, y = _toy()
    m = SoftmaxRegressionNP(lam=0.1, dtype=np.float64)
    Xb, Y = add_bias(X), one_hot(y, 4)
    W = np.random.default_rng(1).normal(0, 0.3, (Xb.shape[1], 4))
    G = m.gradient(Xb, Y, W)
    num = np.zeros_like(W)
    h = 1e-6
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += h
            Wm[i, j] -= h
            num[i, j] = (m.loss(Xb, y, Wp) - m.loss(Xb, y, Wm)) / (2 * h)
    rel = np.linalg.norm(G - num) / (np.linalg.norm(G) + np.linalg.norm(num))
    assert rel < 1e-7, f"sai số tương đối {rel:.2e}"


def check_matches_sklearn_optimum():
    """Cùng hàm mục tiêu: sklearn cực tiểu ½‖w‖² + C Σ CE, tương đương J khi C = 1/(λN)."""
    X, y = _toy()
    lam = 1e-2
    m = SoftmaxRegressionNP(eta=0.5, lam=lam, max_epochs=3000, batch_size=None, momentum=0.9,
                            dtype=np.float64, tol=1e-12).fit(X, y)
    sk = LogisticRegression(C=1 / (lam * len(X)), tol=1e-10, max_iter=10_000).fit(X, y)
    diff = np.abs(m.predict_proba(X) - sk.predict_proba(X)).max()
    assert diff < 1e-4, f"xác suất lệch sklearn {diff:.2e}"
    # sklearn chọn nghiệm có tổng trọng số theo lớp bằng 0; so sau khi trừ trung bình theo lớp
    W_mine = m.weights_ - m.weights_.mean(1, keepdims=True)
    assert np.allclose(W_mine, sk.coef_.T, atol=1e-3)


CHECKS = [
    check_gradient_matches_finite_difference,
    check_matches_sklearn_optimum,
]

if __name__ == "__main__":
    main(CHECKS, "logistic_regression")
