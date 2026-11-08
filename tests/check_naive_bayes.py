"""Kiểm tra src/models/naive_bayes.py: đối chiếu với sklearn.
Chạy: python tests/check_naive_bayes.py   (hoặc python tests/run_all.py)
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.naive_bayes import MultinomialNB as SklearnMNB

from _helpers import main
from src.models.naive_bayes import GaussianNBNP, MultinomialNBNP


def _counts_data(seed=0, n=300, d=20, K=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    rates = rng.uniform(0.2, 3.0, size=(K, d))
    X = rng.poisson(rates[y]).astype(np.float64)
    return X, y


def _intensity_data(seed=1, n=300, d=30, K=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    base = rng.uniform(0.0, 1.0, size=(K, d))
    X = np.clip(base[y] + 0.3 * rng.standard_normal((n, d)), 0.0, 1.0)
    X[:, :3] = 0.0
    return X, y


def check_matches_sklearn_counts():
    X, y = _counts_data()
    ours = MultinomialNBNP(alpha=1.0).fit(X, y)
    ref = SklearnMNB(alpha=1.0).fit(X, y)
    np.testing.assert_allclose(ours.class_log_prior_, ref.class_log_prior_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours.feature_log_prob_, ref.feature_log_prob_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def check_matches_sklearn_intensity():
    X, y = _intensity_data()
    ours = MultinomialNBNP(alpha=0.1).fit(X, y)
    ref = SklearnMNB(alpha=0.1).fit(X, y)
    np.testing.assert_allclose(ours.feature_log_prob_, ref.feature_log_prob_, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def check_gaussian_matches_sklearn():
    """_intensity_data có 3 cột hằng 0 (như dead pixel): chỉ ε giữ σ² > 0."""
    X, y = _intensity_data()
    for vs in (1e-9, 1e-2):
        ours = GaussianNBNP(var_smoothing=vs).fit(X, y)
        ref = SklearnGNB(var_smoothing=vs).fit(X, y)
        np.testing.assert_allclose(ours.epsilon_, ref.epsilon_, rtol=1e-12)
        np.testing.assert_allclose(ours.theta_, ref.theta_, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ours.var_, ref.var_, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ours.decision_function(X), ref.predict_joint_log_proba(X), rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def check_linear_form_matches_joint_log_likelihood():
    """weights_/bias_ phải cho đúng joint log-likelihood của sklearn (dạng tuyến tính ở docstring)."""
    X, y = _intensity_data()
    ours = MultinomialNBNP(0.5).fit(X, y)
    ref = SklearnMNB(alpha=0.5).fit(X, y)
    assert ours.weights_.shape == (X.shape[1], 3) and ours.bias_.shape == (3,)
    np.testing.assert_allclose(ours.decision_function(X), ref.predict_joint_log_proba(X), rtol=1e-10, atol=1e-8)


CHECKS = [
    check_matches_sklearn_counts,
    check_matches_sklearn_intensity,
    check_gaussian_matches_sklearn,
    check_linear_form_matches_joint_log_likelihood,
]

if __name__ == "__main__":
    main(CHECKS, "naive_bayes")
