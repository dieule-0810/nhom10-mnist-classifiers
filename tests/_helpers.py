"""Tiện ích dùng chung cho các file kiểm tra (không cần pytest).

Chạy từng file:   python tests/check_mlp.py
Chạy tất cả:      python tests/run_all.py
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Skip(Exception):
    """Ném ra để bỏ qua một kiểm tra (vd: chưa cài torch)."""


def run_checks(checks) -> tuple[int, int, int]:
    """Chạy danh sách hàm kiểm tra, in kết quả. Trả về (đạt, lỗi, bỏ qua)."""
    passed = failed = skipped = 0
    for check in checks:
        name = check.__name__
        try:
            check()
        except Skip as s:
            skipped += 1
            print(f"  BỎ QUA  {name}: {s}")
        except Exception:
            failed += 1
            print(f"  LỖI     {name}")
            print("          " + traceback.format_exc().strip().replace("\n", "\n          "))
        else:
            passed += 1
            print(f"  ĐẠT     {name}")
    return passed, failed, skipped


def main(checks, title: str) -> None:
    """Điểm vào khi chạy một file kiểm tra trực tiếp."""
    print(f"== {title} ==")
    passed, failed, skipped = run_checks(checks)
    print(f"-- {passed} đạt, {failed} lỗi, {skipped} bỏ qua")
    sys.exit(1 if failed else 0)
