"""Đọc / ghi kết quả theo MỘT cấu trúc chung cho mọi mô hình (xem results/README.md). Chỉ I/O, không vẽ hình.

    results/<key>/log.txt              Logger: mọi bảng script thí nghiệm in ra
    results/<key>/summary.json         write_json: toàn bộ số liệu thí nghiệm, gồm mọi số liệu mà hình cần
    results/<key>/final_metrics.json   save_final_metrics / load_final_metrics: mô hình cuối, cấu trúc chung
    results/<key>/model.pkl            save_model / load_model: mô hình cuối đã huấn luyện
    results/<key>/figures/             do src/plots/<key>.py vẽ từ các file trên
src/experiments/compare_models.py đọc final_metrics.json của mọi mô hình để so sánh.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import BLOCK, SPLITS
from src.evaluation.analysis import block_ids

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
N_BLOCKS = int(block_ids(BLOCK).max() + 1)


# ---------------------------------------------------------------- thư mục, log
def model_dir(key: str) -> Path:
    """results/<key>/, tạo nếu chưa có."""
    path = RESULTS_DIR / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def fig_dir(key: str) -> Path:
    path = model_dir(key) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_path(path) -> str:
    """Đường dẫn tính từ gốc repo (results/...), để log.txt giống nhau trên mọi máy."""
    return Path(path).resolve().relative_to(RESULTS_DIR.parent).as_posix()


def clear_figures(key: str) -> None:
    """Xoá mọi hình trong results/<key>/figures/ trước khi vẽ lại toàn bộ."""
    for old in fig_dir(key).glob("*.png"):
        old.unlink()


class Logger:
    """In ra màn hình và ghi từng dòng vào file (không mất log nếu script dừng giữa chừng).

    Dùng như một hàm: log = Logger(model_dir(key) / "log.txt"); log("...").
    Mỗi dòng được flush ngay; file đóng khi script kết thúc.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


# ---------------------------------------------------------------- JSON
def to_jsonable(obj):
    """numpy -> kiểu Python thường; NaN/inf -> None (JSON không có NaN); bỏ khoá bắt đầu bằng "_"."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def compact_eval(res):
    """Kết quả evaluate() gọn cho JSON: bỏ per_class, giữ ma trận nhầm lẫn và F1 từng lớp."""
    return {s: {k: v for k, v in res[s].items() if k not in ("per_class", "confusion_matrix")}
            | {"confusion_matrix": res[s]["confusion_matrix"].tolist(),
               "per_class_f1": res[s]["per_class"]["f1"].tolist()} for s in SPLITS}


# ---------------------------------------------------------------- final_metrics.json
def save_final_metrics(key, display_name, eval_result, config, *, probabilistic=None,
                       block_importance=None,
                       n_jobs=1, extra=None, quick=False):
    """Ghi results/<key>/final_metrics.json.

    eval_result      : dict từ metrics.evaluate (có "train", "val", "test"); eval_result["fit_time"]
                       theo định nghĩa chung ở results/README.md (KHÔNG gồm chẩn đoán như đường học, OOB).
    probabilistic    : {split: analysis.probabilistic_scores(...)} hoặc None nếu không có predict_proba
    block_importance : 49 phần tử, analysis.block_permutation_importance trên val: số đo CHUNG cho mọi mô hình
    n_jobs           : số luồng dùng khi huấn luyện (để so thời gian công bằng)
    """
    splits = {}
    for s in SPLITS:
        r = eval_result[s]
        cm = np.asarray(r["confusion_matrix"])
        splits[s] = {
            "n": int(cm.sum()),
            "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "f1_macro": r["f1_macro"],
            "predict_time": r.get("predict_time"),
            "per_class": {k: np.asarray(v) for k, v in r["per_class"].items()},
            "confusion_matrix": cm,
            "probabilistic": (probabilistic or {}).get(s),
        }
    if block_importance is not None:
        block_importance = np.asarray(block_importance, dtype=float).ravel()
        if block_importance.size != N_BLOCKS:
            raise ValueError(f"block_importance phải có {N_BLOCKS} phần tử, nhận {block_importance.size}.")
    out = {
        "model": key,
        "display_name": display_name,
        "quick": bool(quick),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "fit_time": eval_result.get("fit_time"),
        "runtime": {"n_jobs": int(os.cpu_count() if n_jobs in (-1, None) else n_jobs),
                    "cpu_count": os.cpu_count()},
        "splits": splits,
        "block_importance": block_importance,
        "block_importance_kind": (f"accuracy giảm khi hoán vị khối {BLOCK}×{BLOCK} trên val"
                                  if block_importance is not None else None),
        "extra": extra or {},
    }
    return write_json(out, model_dir(key) / "final_metrics.json")


def load_final_metrics(key):
    """Đọc results/<key>/final_metrics.json; trả về None nếu chưa có."""
    path = RESULTS_DIR / key / "final_metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- bảng chữ cho log.txt
def format_confusion(cm, title=""):
    """Ma trận nhầm lẫn dạng text để in vào log.txt (hàng: nhãn thật, cột: dự đoán)."""
    cm = np.asarray(cm)
    width = max(5, len(str(cm.max())) + 1)
    lines = [title] if title else []
    lines.append(" thật\\đoán" + "".join(f"{j:>{width}}" for j in range(len(cm))) + f"{'đúng %':>9}")
    for i, row in enumerate(cm):
        acc = row[i] / row.sum() if row.sum() else float("nan")
        lines.append(f"{i:>10}" + "".join(f"{v:>{width}}" for v in row) + f"{acc:>9.1%}")
    return "\n".join(lines)


# ---------------------------------------------------------------- mô hình đã huấn luyện
def model_path(key: str, name: str = "model") -> Path:
    """results/<key>/<name>.pkl: mô hình cuối là model.pkl; thành viên của tổ hợp có tên riêng (cnn_s0, ...)."""
    return RESULTS_DIR / key / f"{name}.pkl"


def save_model(key, model, name="model"):
    """pickle mô hình; bỏ bộ nhớ đệm của lần forward cuối trong các lớp CNN (vài chục MB, không cần để dự đoán)."""
    inner = getattr(model, "model", model)
    for layer in getattr(inner, "layers_", []):
        for attr in ("_cache", "_mask", "_x", "_shape"):
            vars(layer).pop(attr, None)
    path = model_path(key, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(key, name="model"):
    with open(model_path(key, name), "rb") as f:
        return pickle.load(f)


def load_summary(key):
    """Đọc results/<key>/summary.json (toàn bộ số liệu thí nghiệm, gồm số liệu cho hình)."""
    return json.loads((RESULTS_DIR / key / "summary.json").read_text(encoding="utf-8"))
