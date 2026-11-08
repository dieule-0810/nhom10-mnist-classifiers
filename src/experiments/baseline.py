"""Mốc thấp nhất cho mọi mô hình: bộ phân loại tâm gần nhất (mốc trong mọi bảng so sánh, EDA 3.1).

Chạy MỘT lần từ gốc repo (không có siêu tham số, kết quả không đổi giữa các lần chạy):
    python -m src.experiments.baseline

Ghi results/nearest_centroid/ (final_metrics.json, log.txt); hình figures/eval_confusion.png do
src/plots/baseline.py vẽ từ final_metrics.json (gọi ở cuối script).
Chỉ script này ghi các file đó; script mô hình chỉ đọc hoặc tính lại để in bảng, nên kết quả mốc
không phụ thuộc thứ tự chạy các script.
"""

from src.config import BASELINE_KEY, DISPLAY_NAMES, PERM_REPEATS, SEED
from src.data.preprocess import load_splits
from src.evaluation.analysis import block_permutation_importance
from src.evaluation.metrics import evaluate
from src.evaluation.results_io import Logger, format_confusion, model_dir, rel_path, save_final_metrics
from src.models.nearest_centroid import NearestCentroidNP
from src.plots.baseline import plot_all


def main():
    log = Logger(model_dir(BASELINE_KEY) / "log.txt")
    splits = load_splits()
    name = DISPLAY_NAMES[BASELINE_KEY]

    nc = NearestCentroidNP()
    res = evaluate(nc, splits)
    for s in ("train", "val", "test"):
        r = res[s]
        log(f"{s:>5}: acc {r['accuracy']:.4f} | P {r['precision_macro']:.4f} | R {r['recall_macro']:.4f} | "
            f"F1 {r['f1_macro']:.4f} | dự đoán {r['predict_time']:.3f} s")
        log(format_confusion(r["confusion_matrix"], f"[{s}]"))
    log(f"fit {res['fit_time']:.3f} s")

    blocks = block_permutation_importance(nc, *splits["val"], n_repeats=PERM_REPEATS, seed=SEED)
    log(f"Hoán vị khối 4×4 trên val: khối quan trọng nhất làm giảm acc {blocks.max():.4f}")

    path = save_final_metrics(BASELINE_KEY, name, res, {"features": "pixel [0, 1], 784"},
                              block_importance=blocks, n_jobs=1)
    log(f"Đã ghi {rel_path(path)}")
    plot_all()


if __name__ == "__main__":
    main()
