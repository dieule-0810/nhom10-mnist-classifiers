"""Tải MNIST từ Kaggle (hojjatk/mnist-dataset) vào data/raw/.

Chạy từ gốc repo:
    python -m src.data.download            # bỏ qua nếu đã đủ 4 file
    python -m src.data.download --force    # tải lại

File zip trên Kaggle chứa mỗi file hai lần (ở gốc và trong thư mục con cùng tên);
script chỉ giữ 4 file ở gốc rồi đọc thử bằng load_mnist.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.data.idx_loader import DEFAULT_RAW_DIR, _MNIST_FILES, load_mnist

KAGGLE_DATASET = "hojjatk/mnist-dataset"


def _missing_files(raw_dir: Path) -> list[str]:
    return [
        names[0] for names in _MNIST_FILES.values()
        if not any((raw_dir / name).is_file() for name in names)
    ]


def download(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> None:
    if not force and not _missing_files(raw_dir):
        print(f"Đã có đủ 4 file trong {raw_dir}, bỏ qua (dùng --force để tải lại).")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    # Giải nén vào thư mục tạm cạnh data/raw để không lẫn với file cũ.
    with tempfile.TemporaryDirectory(dir=raw_dir.parent) as tmp:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
               "-d", KAGGLE_DATASET, "-p", tmp, "--unzip"]
        print("Đang tải:", " ".join(cmd[1:]))
        if subprocess.run(cmd).returncode != 0:
            sys.exit("Tải thất bại. Kiểm tra mạng, hoặc chạy `kaggle auth login` rồi thử lại.")

        tmp_dir = Path(tmp)
        for names in _MNIST_FILES.values():
            src = next((tmp_dir / n for n in names if (tmp_dir / n).is_file()), None)
            if src is None:
                sys.exit(f"Không thấy {names[0]} trong file tải về.")
            shutil.copyfile(src, raw_dir / names[0])

    (X, _), (Xt, _) = load_mnist(raw_dir)
    print(f"Xong: {raw_dir} — train {X.shape}, test {Xt.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="tải lại dù đã có đủ file")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
