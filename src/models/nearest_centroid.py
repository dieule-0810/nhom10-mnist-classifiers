"""Bộ phân loại tâm gần nhất: mốc thấp nhất (baseline) trong mọi bảng so sánh.

Dùng chung cho mọi mô hình; kết quả ghi một lần bởi src/experiments/baseline.py.

EDA mục 3.1: tâm các lớp tách nhau rõ (khoảng cách Euclid 2,48 đến 7,57), nên "gán ảnh vào
lớp có ảnh trung bình gần nhất" đã là một bộ phân lớp có cơ sở. Mọi mô hình của đồ án phải
vượt mốc này.

Khoảng cách Euclid ‖x − μ_k‖² = ‖x‖² − 2 μ_kᵀx + ‖μ_k‖², và ‖x‖² giống nhau cho mọi lớp, nên
    ŷ = argmax_k [ μ_kᵀ x − ½ ‖μ_k‖² ]
cũng là một bộ phân lớp TUYẾN TÍNH, trọng số w_k = μ_k: cùng dạng hàm với Logistic Regression,
chỉ khác cách chọn W. Ảnh trọng số của nó chính là ảnh trung bình ở EDA 1.2.
"""

import numpy as np

from src.models.base import BaseClassifier


class NearestCentroidNP(BaseClassifier):
    def fit(self, X, y):
        X, y = self._validate_fit(X, y)
        self.centroids_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])   # (K, d)
        self.bias_ = -0.5 * (self.centroids_ ** 2).sum(axis=1)
        return self

    def decision_function(self, X):
        """μ_kᵀx − ½‖μ_k‖² = −½ ‖x − μ_k‖² + hằng số theo x."""
        return self._validate_predict(X) @ self.centroids_.T + self.bias_

    def predict(self, X):
        scores = self.decision_function(X)                  # kiểm tra đã fit trước khi đọc classes_
        return self.classes_[np.argmax(scores, axis=1)]

    @property
    def weights_(self):
        return self.centroids_.T

    def __repr__(self):
        return "NearestCentroidNP()"