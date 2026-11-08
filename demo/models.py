"""Danh sách và cách nạp mô hình cho demo (dùng chung bởi demo/app.py và demo/robustness.py).

Mỗi mô hình đơn là một Pipeline đã lưu ở results/<key>/model.pkl. Riêng "cnn_ensemble" là seed ensemble của
tổ hợp cuối: 5 CNN cùng cấu hình, khác seed, lưu ở results/ensemble/cnn_s<seed>.pkl, dự đoán bằng trung bình
xác suất (soft voting không trọng số, như soft_vote trong src/experiments/ensemble.py). Demo không kèm RF của
tổ hợp: trên test 5 CNN một mình đã bằng phương án có RF (cùng 85 ảnh sai), RF làm xác suất kém đi (log-loss
0,041 so với 0,025), và file RF (~220 MB) không có trong Git.
"""

import numpy as np

from src.evaluation.results_io import load_final_metrics, load_model, model_path

ENSEMBLE_KEY = "cnn_ensemble"
DEMO_MODELS = ["cnn", ENSEMBLE_KEY, "mlp", "random_forest", "logistic_regression", "naive_bayes"]


class SeedEnsemble:
    """Trung bình xác suất của nhiều Pipeline (cùng giao diện predict / predict_proba)."""

    def __init__(self, members):
        self.members = list(members)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def _ensemble_files():
    """Tên file của 5 CNN, theo danh sách seed ghi trong results/ensemble/final_metrics.json."""
    fm = load_final_metrics("ensemble")
    if fm is None:
        return []
    return [f"cnn_s{name.removeprefix('CNN-s')}" for name in fm["config"]["cnn_seeds"]]


def demo_paths(key):
    """Các file .pkl mà mô hình `key` cần."""
    if key == ENSEMBLE_KEY:
        return [model_path("ensemble", f) for f in _ensemble_files()]
    return [model_path(key)]


def is_available(key):
    paths = demo_paths(key)
    return bool(paths) and all(p.is_file() for p in paths)


def load_demo_model(key):
    if key == ENSEMBLE_KEY:
        return SeedEnsemble(load_model("ensemble", f) for f in _ensemble_files())
    return load_model(key)
