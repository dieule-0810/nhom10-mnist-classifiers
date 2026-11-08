"""Interface chung cho mọi bộ phân lớp của nhóm.

Quy ước (giống sklearn, để đối chiếu chỉ cần đổi một dòng):
    - X: np.ndarray float, shape (n_samples, n_features), thường là (N, 784).
    - y: np.ndarray int, shape (n_samples,), giá trị 0..n_classes-1.
    - fit(X, y)          -> trả về self (để viết được model.fit(X, y).predict(X)).
    - predict(X)         -> np.ndarray int, shape (n_samples,).
    - predict_proba(X)   -> np.ndarray float, shape (n_samples, n_classes), mỗi hàng tổng = 1.
                            Không bắt buộc; mô hình nào không có thì để NotImplementedError.
    - Siêu tham số truyền qua __init__ và lưu nguyên tên thuộc tính,
      để get_params() tự lấy ra khi ghi log thí nghiệm.
    - Thuộc tính học được sau fit có hậu tố "_" (vd: self.weights_, self.classes_).
    - Mọi thành phần ngẫu nhiên nhận tham số seed (mặc định src.config.SEED) để tái lập kết quả.
    - Mô hình cần tiền xử lý riêng thì ghép bằng src.models.pipeline.Pipeline, không tự làm trong fit.

Ví dụ:
    class LogisticRegressionGD(BaseClassifier):
        def __init__(self, lr=0.1, n_iters=100, seed=0):
            self.lr = lr
            self.n_iters = n_iters
            self.seed = seed

        def fit(self, X, y):
            X, y = self._validate_fit(X, y)
            ...
            return self

        def predict(self, X):
            X = self._validate_predict(X)
            ...
"""

import inspect
from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} không hỗ trợ predict_proba.")

    def get_params(self) -> dict:
        """Lấy siêu tham số từ chữ ký __init__ (dùng để ghi log thí nghiệm)."""
        names = [
            p for p in inspect.signature(type(self).__init__).parameters
            if p != "self"
        ]
        return {name: getattr(self, name) for name in names if hasattr(self, name)}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    # --- Kiểm tra đầu vào dùng chung ---

    def _validate_fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X phải 2 chiều (n_samples, n_features), nhận shape {X.shape}.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y phải có shape ({len(X)},), nhận {y.shape}.")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return X, y.astype(np.int64)

    def _validate_predict(self, X):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{type(self).__name__} chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X cần shape (n, {self.n_features_in_}), nhận {X.shape}."
            )
        return X
