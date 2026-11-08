"""Tiền xử lý dùng chung cho mọi mô hình.

Chỉ gồm các bước "cơ khí", không chứa quyết định phụ thuộc EDA:
    1. Đưa về float32 trong [0, 1] (chia cho 255) - chỉ đổi thang đo, không mất thông tin.
    2. Làm phẳng 28x28 -> 784.
    3. Tách validation từ tập train (stratified, seed cố định).

Các bước tùy mô hình (standardize, bỏ dead pixel, lượng tử, ...) KHÔNG đặt ở đây;
chúng được quyết định sau EDA và tham số phải được fit trên tập train.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import SPLIT_SEED, VAL_SIZE
from src.data.idx_loader import load_mnist

DEFAULT_VAL_SIZE = VAL_SIZE
DEFAULT_SEED = SPLIT_SEED


def to_unit_range(X: np.ndarray) -> np.ndarray:
    """uint8 0..255 -> float32 0..1. Ép kiểu trước khi chia để tránh tràn số uint8."""
    return X.astype(np.float32) / 255.0


def flatten(X: np.ndarray) -> np.ndarray:
    """(N, 28, 28) -> (N, 784). Giữ nguyên thứ tự pixel theo hàng."""
    return X.reshape(len(X), -1)


def stratified_split(
    y: np.ndarray, val_size: int = DEFAULT_VAL_SIZE, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (train_idx, val_idx) sao cho tỷ lệ các lớp trong val giống tập gốc.

    Dùng train_test_split(stratify=y) của sklearn; chỉ số được sắp lại để giữ thứ tự ảnh gốc.
    """
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=val_size,
                                          stratify=y, random_state=seed)
    return np.sort(train_idx), np.sort(val_idx)


def subset_train(splits: dict, n: int, seed: int = DEFAULT_SEED) -> dict:
    """Bản sao của splits với tập train chỉ còn n ảnh chọn ngẫu nhiên (dùng cho chế độ --quick)."""
    X, y = splits["train"]
    idx = np.random.default_rng(seed).permutation(len(X))[:n]
    return dict(splits, train=(X[idx], y[idx]))


def load_splits(
    raw_dir: str | Path | None = None,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    flat: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Điểm vào chung cho mọi người trong nhóm.

    Trả về dict {"train": (X, y), "val": (X, y), "test": (X, y)}.
        flat=True : X có shape (N, 784)    -> dùng cho mô hình
        flat=False: X có shape (N, 28, 28) -> dùng cho EDA, vẽ ảnh
    X là float32 trong [0, 1]; y là int64 trong 0..9.

    Cùng val_size và seed thì cách chia luôn giống nhau trên mọi máy.
    """
    (X_tr_full, y_tr_full), (X_test, y_test) = load_mnist(raw_dir)
    train_idx, val_idx = stratified_split(y_tr_full, val_size, seed)

    def _prep(X, y):
        X = to_unit_range(X)
        return (flatten(X) if flat else X), y.astype(np.int64)

    return {
        "train": _prep(X_tr_full[train_idx], y_tr_full[train_idx]),
        "val": _prep(X_tr_full[val_idx], y_tr_full[val_idx]),
        "test": _prep(X_test, y_test),
    }
