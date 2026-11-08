"""Chạy toàn bộ kiểm tra.  Chạy: python tests/run_all.py

Tự tìm mọi file tests/check_*.py có danh sách CHECKS, nên thêm file kiểm tra mới
không cần sửa file này.
Chạy một phần: python tests/run_all.py cnn mlp
"""

import importlib
import sys
from pathlib import Path

from _helpers import run_checks

HERE = Path(__file__).resolve().parent


def discover(filters=()):
    names = sorted(p.stem for p in HERE.glob("check_*.py"))
    if filters:
        names = [n for n in names if any(f in n for f in filters)]
    return names


def main():
    total_pass = total_fail = total_skip = 0
    for name in discover(sys.argv[1:]):
        print(f"== {name} ==")
        module = importlib.import_module(name)
        checks = getattr(module, "CHECKS", None)
        if checks is None:
            print(f"  LỖI    {name}.py không có danh sách CHECKS")
            total_fail += 1
            continue
        p, f, s = run_checks(checks)
        total_pass, total_fail, total_skip = total_pass + p, total_fail + f, total_skip + s

    print(f"\nTỔNG: {total_pass} đạt, {total_fail} lỗi, {total_skip} bỏ qua")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
