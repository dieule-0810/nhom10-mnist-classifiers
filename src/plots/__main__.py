"""Vẽ lại mọi hình từ kết quả đã lưu: python -m src.plots [key ...] (mặc định: mọi mô hình đã có kết quả)."""

import importlib
import sys

from src.config import BASELINE_KEY, ENSEMBLE_KEY, MODEL_KEYS
from src.evaluation.results_io import RESULTS_DIR

# key thư mục results/ -> module trong src/plots/
MODULES = {BASELINE_KEY: "baseline", **{k: k for k in MODEL_KEYS}, ENSEMBLE_KEY: "ensemble",
           "comparison": "compare_models", "demo": "demo"}


def main(keys=None):
    for key in keys or MODULES:
        if key not in MODULES:
            print(f"{key}: không có hình để vẽ (các khoá: {', '.join(MODULES)})")
            continue
        if not (RESULTS_DIR / key / "final_metrics.json").is_file() and not (RESULTS_DIR / key / "summary.json").is_file():
            print(f"{key}: chưa có kết quả, bỏ qua")
            continue
        module = importlib.import_module(f"src.plots.{MODULES[key]}")
        for path in module.plot_all():
            print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
