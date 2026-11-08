"""Hàm số học dùng chung cho các mô hình có hàm softmax / log-xác suất (LR, MLP, Naive Bayes).

Đặt riêng một file vì nhiều mô hình cần đúng các hàm này: LR và MLP (lớp ra softmax,
cross-entropy), Naive Bayes cần log-sum-exp để chuẩn hoá log P(x, y) thành P(y | x).
"""

import numpy as np


def add_bias(X):
    """Thêm cột 1 vào cuối ma trận: x̃ = [x; 1]."""
    return np.ascontiguousarray(np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)]))


def logsumexp(Z, axis=1, keepdims=True):
    """log Σ exp(z) theo hàng, trừ max trước khi lấy exp để tránh tràn số."""
    m = Z.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(Z - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def log_softmax(Z):
    """log softmax theo hàng."""
    return Z - logsumexp(Z)


def softmax(Z):
    return np.exp(log_softmax(Z))


def one_hot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y
