"""Hình các bước chuẩn hoá ảnh vẽ của demo, vẽ từ results/demo/summary.json (do demo/robustness.py ghi).

    demo_pipeline.png   (a) ảnh vẽ 280×280 → (b) cắt khung chữ → (c) thu nhỏ vào 20×20 → (d) căn tâm khối lượng
                        trong 28×28 → (e) xác suất của lớp được đoán ở từng mô hình

Các ảnh (a)–(d) được dựng lại từ ảnh test và độ lệch đã lưu (demo/drawing.py), không cần mô hình.
Chạy: python -m src.plots.demo   (hoặc python -m src.plots để vẽ mọi mô hình)
"""

import matplotlib.pyplot as plt
import numpy as np

from demo.drawing import drawn_canvas, mnist_steps
from src.config import DISPLAY_NAMES, MODEL_COLORS
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_summary
from src.plots.common import savefig

KEY = "demo"


def plot_pipeline(summary, X_te, path):
    ex = summary["example"]
    steps = mnist_steps(drawn_canvas(X_te[ex["test_index"]], ex["shift"], summary["scale"]))
    panels = [(f"(a) Ảnh vẽ {steps['ink'].shape[0]}×{steps['ink'].shape[1]}", steps["ink"]),
              (f"(b) Cắt khung chữ {steps['crop'].shape[0]}×{steps['crop'].shape[1]}", steps["crop"]),
              (f"(c) Thu nhỏ {steps['small'].shape[0]}×{steps['small'].shape[1]}", steps["small"]),
              (f"(d) Căn tâm khối lượng {steps['out'].shape[0]}×{steps['out'].shape[1]}", steps["out"])]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.4]))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    ax = axes[4]
    keys = list(ex["proba"])[::-1]
    P = np.array([ex["proba"][k] for k in keys])
    pred, conf = P.argmax(1), P.max(1)
    ax.barh(range(len(keys)), conf, color=[MODEL_COLORS[k] for k in keys])
    for i, (d, c) in enumerate(zip(pred, conf)):
        ax.text(c - 0.02, i, f"đoán {d} ({c:.2f})", va="center", ha="right", color="white", fontsize=9)
    ax.set_yticks(range(len(keys)), [DISPLAY_NAMES[k] for k in keys])
    ax.set_xlim(0, 1)
    ax.set_xlabel("xác suất của lớp được đoán")
    ax.set_title("(e) Dự đoán của các mô hình", fontsize=10)
    fig.tight_layout()
    return savefig(fig, path)


def plot_all():
    summary = load_summary(KEY)
    X_te = load_splits()["test"][0]
    clear_figures(KEY)
    return [plot_pipeline(summary, X_te, fig_dir(KEY) / "demo_pipeline.png")]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
