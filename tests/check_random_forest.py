"""Kiểm tra các hàm phân tích rừng trong src/models/random_forest.py, đối chiếu cách tính độc lập.
Chạy: python tests/check_random_forest.py
"""

import numpy as np

from _helpers import main
from src.models.random_forest import make_random_forest, staged_accuracy, strength_correlation, tree_votes


def _make_codes(n=600, d=12, n_bins=16, n_classes=3, seed=0):
    """Dữ liệu tổng hợp dạng mã bin, nhãn phụ thuộc phi tuyến vào vài cột."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_bins, size=(n, d)).astype(np.uint8)
    score = X[:, 0].astype(int) + 2 * (X[:, 1] > 8) - (X[:, 2] > 12) * X[:, 3] / 4
    y = np.digitize(score + rng.normal(0, 1.5, n), np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))
    return X, y.astype(np.int64)


def _forest(n_estimators=15):
    X, y = _make_codes()
    return make_random_forest(n_estimators=n_estimators, n_jobs=1, seed=1).fit(X, y), X, y


def check_soft_votes_average_to_predict_proba():
    """Soft voting: trung bình phân phối lớp ở lá của các cây đúng bằng rf.predict_proba."""
    rf, X, _ = _forest()
    V = tree_votes(rf, X, "soft")
    assert V.shape == (rf.n_estimators, len(X), 3), V.shape
    assert np.allclose(V.mean(axis=0), rf.predict_proba(X), atol=1e-12)


def check_hard_votes_are_one_hot():
    rf, X, _ = _forest()
    V = tree_votes(rf, X, "hard")
    assert np.array_equal(V.sum(axis=2), np.ones(V.shape[:2]))
    expected = np.stack([t.predict_proba(X).argmax(axis=1) for t in rf.estimators_])
    assert np.array_equal(V.argmax(axis=2), expected)


def check_staged_accuracy_matches_forests_of_each_size():
    """Phần tử thứ t là accuracy của rừng gồm t cây đầu; phần tử cuối bằng accuracy của rf.predict."""
    rf, X, y = _forest()
    Xt, yt = _make_codes(seed=9)
    acc = staged_accuracy(rf, Xt, yt, "soft")
    assert len(acc) == rf.n_estimators
    assert np.isclose(acc[-1], (rf.predict(Xt) == yt).mean())
    for t in (1, 5, 10):
        P = np.mean([e.predict_proba(Xt) for e in rf.estimators_[:t]], axis=0)
        assert np.isclose(acc[t - 1], (P.argmax(axis=1) == yt).mean()), t


def check_strength_correlation_matches_pairwise_loop():
    rf, X, y = _forest(n_estimators=8)
    strength, corr = strength_correlation(rf, X, y)
    errors = [(t.predict_proba(X).argmax(axis=1) != y).astype(float) for t in rf.estimators_]
    assert np.isclose(strength, np.mean([1 - e.mean() for e in errors]))
    pairs = [np.corrcoef(errors[i], errors[j])[0, 1] for i in range(len(errors)) for j in range(i + 1, len(errors))]
    assert np.isclose(corr, np.nanmean(pairs))


def check_invalid_voting_raises():
    rf, X, _ = _forest(n_estimators=2)
    try:
        tree_votes(rf, X, "majority")
    except ValueError:
        return
    raise AssertionError("voting không hợp lệ phải báo ValueError")


CHECKS = [
    check_soft_votes_average_to_predict_proba,
    check_hard_votes_are_one_hot,
    check_staged_accuracy_matches_forests_of_each_size,
    check_strength_correlation_matches_pairwise_loop,
    check_invalid_voting_raises,
]

if __name__ == "__main__":
    main(CHECKS, "random_forest")
