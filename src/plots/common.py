"""Hàm vẽ dùng chung cho mọi mô hình. Chỉ đọc số liệu đã lưu, không huấn luyện.

Mọi hình nằm trong results/<key>/figures/ và được vẽ từ results/<key>/summary.json, final_metrics.json
(src/plots/<key>.py). Hai hình có tên chung cho mọi mô hình:
    eval_confusion.png          plot_confusion_splits (bắt buộc)
    eval_confident_errors.png   plot_confident_errors (mô hình có predict_proba)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EDA_CLOSE_PAIRS, IMAGE_SHAPE, SPLITS

# Màu cho các đường trong cùng một hình (khi các đường không phải là các mô hình khác nhau)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c43c5c", "#6b6b6b"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def savefig(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_splits(splits_metrics, path, title, highlight_pairs=EDA_CLOSE_PAIRS):
    """Hình 3 ô train | val | test. Màu theo tỉ lệ trong từng hàng (recall), số trong ô là số ảnh.

    splits_metrics: {tập: {"confusion_matrix": ...}}, ví dụ final_metrics["splits"].
    Tô màu theo tỉ lệ hàng thay vì số đếm, vì 3 tập có kích thước khác nhau (50k / 10k / 10k).
    highlight_pairs: các cặp (i, j) được đóng khung (mặc định 5 cặp EDA 3.1); None để tắt.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, s in zip(axes, SPLITS):
        cm = np.asarray(splits_metrics[s]["confusion_matrix"])
        rate = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(np.sqrt(rate), cmap="Blues", vmin=0, vmax=1)     # căn bậc hai để thấy rõ ô lỗi nhỏ
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if rate[i, j] > 0.5 else "black")
        for i, j in highlight_pairs or []:
            for a, b in [(i, j), (j, i)]:
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="#e8833a", lw=1.8))
        acc = np.trace(cm) / cm.sum()
        ax.set(xticks=range(10), yticks=range(10), xlabel="dự đoán", ylabel="nhãn thật",
               title=f"{s} (n = {cm.sum():,}; accuracy = {acc:.4f})".replace(",", "."))   # 50.000 kiểu Việt
    note = " Khung cam: 5 cặp EDA 3.1." if highlight_pairs else ""
    fig.suptitle(f"{title}: ma trận nhầm lẫn (màu theo tỉ lệ trong hàng, số là số ảnh).{note}", fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def plot_confident_errors(X, y, records, path, title):
    """Các ảnh bị đoán sai với xác suất cao nhất (2 hàng). Tên chung: eval_confident_errors.png.

    records: {"idx": chỉ số ảnh trong (X, y), "proba": hàng xác suất tương ứng}, từ
    src.evaluation.analysis.confident_error_records, đã sắp theo xác suất giảm dần.
    """
    idx = np.asarray(records["idx"], dtype=int)
    proba = np.asarray(records["proba"], dtype=float)
    fig, axes = plt.subplots(2, int(np.ceil(max(len(idx), 2) / 2)), figsize=(14, 3.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i, p in zip(axes.ravel(), idx, proba):
        pred = int(p.argmax())
        ax.imshow(np.asarray(X[i]).reshape(IMAGE_SHAPE), cmap="gray_r")
        ax.set_title(f"thật {y[i]}, đoán {pred}\np = {p[pred]:.2f}", fontsize=8)
    fig.suptitle(f"{title}: {len(idx)} ảnh bị đoán sai với xác suất cao nhất")
    plt.tight_layout()
    return savefig(fig, path)


def plot_importance_maps(panels, path, suptitle="", cmap="magma"):
    """Nhiều bản đồ 28×28 cạnh nhau. panels: list (tiêu đề, mảng 784 hoặc 28×28); NaN tô xám."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4), squeeze=False)
    for ax, (title, M) in zip(axes[0], panels):
        cm_ = plt.get_cmap(cmap).copy()
        cm_.set_bad("#9e9e9e")
        im = ax.imshow(np.ma.masked_invalid(np.asarray(M, dtype=float).reshape(IMAGE_SHAPE)), cmap=cm_)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 7, 14, 21, 27]); ax.set_yticks([0, 7, 14, 21, 27])
        fig.colorbar(im, ax=ax, fraction=0.046)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def nan_array(values):
    """list từ JSON (None ở chỗ NaN) -> mảng float có NaN."""
    return np.array([np.nan if v is None else v for v in np.asarray(values, dtype=object).ravel()],
                    dtype=float).reshape(np.shape(values))
