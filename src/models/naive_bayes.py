"""Naive Bayes tự cài bằng NumPy: Multinomial (mô hình chính, EDA 5.1) và Gaussian (đối chứng ở E3).

Quy tắc MAP với giả định độc lập có điều kiện khi biết lớp, tính trong miền log:

    Multinomial:  log P(c | x) ∝ log π_c + Σ_j x_j log θ_cj
                  θ_cj = (N_cj + α) / (N_c + α·d),   N_cj = tổng cường độ pixel j của lớp c
    Gaussian:     log P(c | x) ∝ log π_c − ½ Σ_j [log(2π σ²_cj) + (x_j − μ_cj)² / σ²_cj]
                  μ_cj, σ²_cj = trung bình, phương sai pixel j trong lớp c; σ²_cj cộng thêm
                  ε = var_smoothing · max_j Var(x_j) (như sklearn)

Multinomial TUYẾN TÍNH theo đặc trưng: điểm của lớp c là w_cᵀx + b_c, giống Logistic Regression và tâm
gần nhất (EDA 5.1, bảng so sánh "chỉ đổi một yếu tố"). Vì vậy nó có weights_ (d, K) và bias_ (K,),
nên Pipeline.pixel_weights() vẽ được ảnh trọng số như với LR.
    Multinomial: w_cj = log θ_cj,   b_c = log π_c
Gaussian là hàm BẬC HAI của x (σ²_cj khác nhau giữa các lớp), nên không có weights_ / bias_.

α > 0 là bắt buộc: 116 pixel gần như chết (EDA 1.1) có tổng mực bằng 0 ở nhiều lớp, α = 0 cho log 0 = −∞.
var_smoothing > 0 là bắt buộc: 72 dead pixel (EDA 1.1) có phương sai 0 trong mọi lớp.
Đối chiếu sklearn.naive_bayes.MultinomialNB / GaussianNB trong tests/check_naive_bayes.py.
"""

import time

import numpy as np

from src.models._numeric import log_softmax
from src.models.base import BaseClassifier


class _BaseNB(BaseClassifier):
    """Phần chung: prior của lớp, dự đoán từ decision_function (joint log-likelihood)."""

    def _class_log_prior(self, y):
        self.class_count_ = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

    def predict(self, X):
        scores = self.decision_function(X)                  # kiểm tra đã fit trước khi đọc classes_
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Chuẩn hoá trong miền log (log-sum-exp) để không tràn số khi cộng 784 số hạng."""
        return np.exp(log_softmax(self.decision_function(X)))


class MultinomialNBNP(_BaseNB):
    """Multinomial Naive Bayes, làm trơn Laplace / Dirichlet α. Đầu vào: cường độ không âm (pixel [0, 1])."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        if self.alpha <= 0:
            raise ValueError(f"alpha phải > 0, nhận {self.alpha}")
        X, y = self._validate_fit(X, y)
        if np.any(X < 0):
            raise ValueError("MultinomialNBNP yêu cầu X >= 0 (non-negative counts)")
        t0 = time.perf_counter()
        self._class_log_prior(y)
        self.feature_count_ = np.stack([X[y == c].sum(axis=0) for c in self.classes_])          # (K, d)
        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))       # log θ, (K, d)
        self.weights_ = self.feature_log_prob_.T
        self.bias_ = self.class_log_prior_
        self.fit_time_ = time.perf_counter() - t0
        return self

    def decision_function(self, X):
        """Joint log-likelihood log P(c) + log P(x | c), shape (n, K)."""
        X = self._validate_predict(X)
        return X @ self.weights_ + self.bias_


class GaussianNBNP(_BaseNB):
    """Gaussian Naive Bayes: mỗi pixel trong mỗi lớp là một phân phối chuẩn N(μ_cj, σ²_cj + ε)."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        if self.var_smoothing <= 0:
            raise ValueError(f"var_smoothing phải > 0, nhận {self.var_smoothing}")
        X, y = self._validate_fit(X, y)
        t0 = time.perf_counter()
        self._class_log_prior(y)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])                  # μ, (K, d)
        self.var_ = np.stack([X[y == c].var(axis=0) for c in self.classes_]) + self.epsilon_      # σ², (K, d)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def decision_function(self, X):
        """Joint log-likelihood log P(c) + Σ_j log N(x_j; μ_cj, σ²_cj), shape (n, K)."""
        X = self._validate_predict(X)
        inv = 1.0 / self.var_                                                                    # (K, d)
        # Σ_j (x_j − μ_cj)² / σ²_cj = x² · (1/σ²) − 2 x · (μ/σ²) + Σ μ²/σ², tính bằng nhân ma trận
        quad = (X ** 2) @ inv.T - 2.0 * X @ (self.theta_ * inv).T + (self.theta_ ** 2 * inv).sum(axis=1)
        log_norm = -0.5 * np.log(2.0 * np.pi * self.var_).sum(axis=1)
        return self.class_log_prior_ + log_norm - 0.5 * quad
