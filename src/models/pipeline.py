"""Pipeline duy nhất của nhóm: biến đổi đặc trưng (src/data/features.py) + một bộ phân lớp.

Dùng được như một mô hình (fit / predict / predict_proba), nên evaluate() trong
src/evaluation/metrics.py đo thời gian huấn luyện và dự đoán GỒM cả tiền xử lý,
và block_permutation_importance nhận thẳng ảnh gốc 784 pixel.
"""

import inspect

import numpy as np

from src.config import IMAGE_SHAPE


class Pipeline:
    def __init__(self, features, model, name=None):
        self.features = features
        self.model = model
        self.name = name or type(model).__name__

    def fit(self, X, y, X_val=None, y_val=None):
        """Học tham số tiền xử lý trên X rồi huấn luyện mô hình.

        X_val, y_val chỉ được chuyển cho mô hình nếu fit của nó nhận hai tham số này
        (early stopping của LR, MLP); val được biến đổi bằng tham số học từ X.
        """
        Z = self.features.fit_transform(X)
        accepts_val = "X_val" in inspect.signature(self.model.fit).parameters
        if X_val is not None and accepts_val:
            self.model.fit(Z, y, X_val=self.features.transform(X_val), y_val=y_val)
        else:
            self.model.fit(Z, y)
        return self

    def predict(self, X):
        return self.model.predict(self.features.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.features.transform(X))

    def decision_function(self, X):
        return self.model.decision_function(self.features.transform(X))

    def pixel_weights(self):
        """Ảnh trọng số theo pixel gốc (K, 28, 28), NaN ở pixel bị bỏ. Chỉ cho mô hình tuyến tính
        (có weights_ shape (d, K)) và đặc trưng có backproject."""
        W = self.features.backproject(np.asarray(self.model.weights_))
        return W.T.reshape(-1, *IMAGE_SHAPE)

    def __repr__(self):
        return f"{self.features!r} -> {self.model!r}"
