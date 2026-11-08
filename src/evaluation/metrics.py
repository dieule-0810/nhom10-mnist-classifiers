"""Độ đo đánh giá: tính bằng sklearn.metrics, gói lại theo một dạng chung cho mọi mô hình.

Đề bài yêu cầu: accuracy, precision, recall, F1 (macro), confusion matrix
trên cả train, validation và test; kèm thời gian huấn luyện và dự đoán.
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import N_CLASSES


def classification_scores(y_true, y_pred, n_classes: int = N_CLASSES) -> dict:
    """Toàn bộ độ đo cho một tập.

    labels = 0..n_classes−1 để ma trận nhầm lẫn và trung bình macro luôn đủ mọi lớp, kể cả lớp vắng mặt
    trong y_true lẫn y_pred; zero_division=0: lớp không bao giờ được dự đoán có precision 0.
    """
    labels = np.arange(n_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {"precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate(model, splits: dict, fit: bool = True) -> dict:
    """Huấn luyện (nếu fit=True) trên splits["train"] và đánh giá trên mọi tập.

    Dùng được cho cả mô hình tự cài lẫn mô hình sklearn, vì cả hai có fit/predict.

    Trả về {"fit_time": giây, "train": {...}, "val": {...}, "test": {...}},
    mỗi tập có thêm "predict_time" (giây).
    """
    result = {"model": type(model).__name__}

    if fit:
        X_tr, y_tr = splits["train"]
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        result["fit_time"] = time.perf_counter() - start

    for name, (X, y) in splits.items():
        start = time.perf_counter()
        y_pred = model.predict(X)
        elapsed = time.perf_counter() - start
        scores = classification_scores(y, y_pred)
        scores["predict_time"] = elapsed
        result[name] = scores
    return result


def summary_table(results: list[dict], split: str = "test") -> str:
    """Bảng so sánh nhiều mô hình trên một tập, dạng text."""
    header = f"{'Mô hình':<24}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Fit(s)':>10}{'Pred(s)':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        s = r[split]
        fit_time = r.get("fit_time", float("nan"))
        lines.append(
            f"{r['model']:<24}{s['accuracy']:>8.4f}{s['precision_macro']:>8.4f}"
            f"{s['recall_macro']:>8.4f}{s['f1_macro']:>8.4f}{fit_time:>10.2f}{s['predict_time']:>10.3f}"
        )
    return "\n".join(lines)
