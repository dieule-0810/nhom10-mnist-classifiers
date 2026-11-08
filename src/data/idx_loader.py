"""Đọc dữ liệu MNIST ở định dạng IDX.

Đặc tả định dạng IDX (http://yann.lecun.com/exdb/mnist/):

    offset  kiểu     ý nghĩa
    0       1 byte   luôn bằng 0
    1       1 byte   luôn bằng 0
    2       1 byte   mã kiểu dữ liệu (0x08 = unsigned byte, ...)
    3       1 byte   số chiều của mảng (ndim)
    4       4*ndim   kích thước từng chiều, số nguyên 32-bit big-endian
    ...              dữ liệu, xếp theo thứ tự hàng (C order)

Với MNIST:
    - file ảnh  : magic = 0x00000803 (2051) -> uint8, 3 chiều (N, 28, 28)
    - file nhãn : magic = 0x00000801 (2049) -> uint8, 1 chiều (N,)
"""

import struct
from pathlib import Path

import numpy as np

from src.config import IMAGE_SHAPE, N_CLASSES

# Mã kiểu dữ liệu theo đặc tả IDX. Các kiểu nhiều byte đều là big-endian.
_IDX_DTYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}

# Thư mục data/raw tính từ gốc repo (src/data/idx_loader.py -> lên 2 cấp).
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Tên file chuẩn. Một số bản trên Kaggle dùng dấu "-" thay cho "." nên chấp nhận cả hai.
_MNIST_FILES = {
    "train_images": ("train-images.idx3-ubyte", "train-images-idx3-ubyte"),
    "train_labels": ("train-labels.idx1-ubyte", "train-labels-idx1-ubyte"),
    "test_images": ("t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte"),
    "test_labels": ("t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte"),
}


def read_idx(path: str | Path) -> np.ndarray:
    """Đọc một file IDX bất kỳ và trả về mảng numpy (ghi được, byte order của máy)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file IDX: {path}")

    data = path.read_bytes()
    if len(data) < 4:
        raise ValueError(f"{path.name}: file quá ngắn, không có header.")

    zero1, zero2, type_code, ndim = struct.unpack(">BBBB", data[:4])
    if zero1 != 0 or zero2 != 0:
        raise ValueError(
            f"{path.name}: magic number không hợp lệ (2 byte đầu phải bằng 0). "
            "File có thể bị nén (.gz) hoặc không phải định dạng IDX."
        )
    if type_code not in _IDX_DTYPES:
        raise ValueError(f"{path.name}: mã kiểu dữ liệu không hỗ trợ: 0x{type_code:02X}")

    header_end = 4 + 4 * ndim
    if len(data) < header_end:
        raise ValueError(f"{path.name}: header bị cắt cụt.")
    dims = struct.unpack(f">{ndim}I", data[4:header_end])

    dtype = _IDX_DTYPES[type_code]
    expected_bytes = int(np.prod(dims, dtype=np.int64)) * dtype.itemsize
    payload = data[header_end:]
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{path.name}: kích thước dữ liệu không khớp header "
            f"(header cần {expected_bytes} byte, thực tế {len(payload)} byte)."
        )

    arr = np.frombuffer(payload, dtype=dtype).reshape(dims)
    # frombuffer trả về mảng chỉ đọc; astype tạo bản sao ghi được, theo byte order của máy.
    return arr.astype(dtype.newbyteorder("="), copy=True)


def _find_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = raw_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Không tìm thấy {candidates[0]} trong {raw_dir}. "
        "Xem mục 'Dữ liệu' trong README để tải MNIST."
    )


def load_mnist(raw_dir: str | Path | None = None):
    """Đọc 4 file MNIST và kiểm tra tính hợp lệ.

    Trả về:
        (X_train, y_train), (X_test, y_test)
        X: uint8, shape (N, 28, 28), giá trị 0..255
        y: uint8, shape (N,), giá trị 0..9
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    arrays = {key: read_idx(_find_file(raw_dir, names)) for key, names in _MNIST_FILES.items()}

    for split in ("train", "test"):
        X, y = arrays[f"{split}_images"], arrays[f"{split}_labels"]
        if X.ndim != 3 or X.shape[1:] != IMAGE_SHAPE:
            raise ValueError(f"Ảnh {split} có shape {X.shape}, kỳ vọng (N, 28, 28).")
        if y.ndim != 1:
            raise ValueError(f"Nhãn {split} có shape {y.shape}, kỳ vọng (N,).")
        if len(X) != len(y):
            raise ValueError(f"Tập {split}: {len(X)} ảnh nhưng {len(y)} nhãn.")
        if y.min() < 0 or y.max() >= N_CLASSES:
            raise ValueError(f"Nhãn {split} nằm ngoài khoảng 0..9.")

    return (
        (arrays["train_images"], arrays["train_labels"]),
        (arrays["test_images"], arrays["test_labels"]),
    )
