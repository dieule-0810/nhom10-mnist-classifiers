"""Hình của Naive Bayes, vẽ từ results/naive_bayes/summary.json và final_metrics.json.

Chạy: python -m src.plots.naive_bayes   (hoặc python -m src.plots để vẽ mọi mô hình)

    hp_e1_alpha.png            E1: val accuracy, log-loss, tỉ lệ lỗi có p > 0,99 theo α
    hp_e3_input.png            E3: cách biểu diễn pixel và dạng phân phối
    interp_theta.png           θ của 10 lớp trên lưới 28×28
    eval_confusion.png         ma trận nhầm lẫn train | val | test
    eval_confident_errors.png  20 ảnh test bị đoán sai tự tin nhất
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import EDA_CLOSE_PAIRS, IMAGE_SHAPE
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics, load_summary
from src.plots.common import nan_array, plot_confident_errors, plot_confusion_splits, savefig

KEY = "naive_bayes"


def plot_e1(rows, best, path):
    """rows: {α (chuỗi): số liệu val}; best: α đã chọn."""
    al = [float(a) for a in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, key, label in [(axes[0], "val_acc", "val accuracy"), (axes[1], "val_loss", "val log-loss"),
                           (axes[2], "overconfident_share_of_errors", "tỉ lệ lỗi có p > 0,99")]:
        ax.plot(al, [r[key] for r in rows.values()], "o-")
        ax.axvline(best, ls=":", c="gray", lw=0.8)
        ax.set(xscale="log", xlabel="α (thang log)", ylabel=label, title=label)
    fig.suptitle("E1: Multinomial NB theo α (validation)", fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def plot_e3(rows, path):
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, key, label in [(axes[0], "val_acc", "val accuracy"), (axes[1], "val_loss", "val log-loss")]:
        ax.barh(range(len(names)), [rows[n][key] for n in names], color=["C0", "C1", "C2", "C3"])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set(xlabel=label, title=label)
        if key == "val_acc":
            lo = min(rows[n][key] for n in names)
            ax.set_xlim(lo - 0.01, max(rows[n][key] for n in names) + 0.005)
    fig.suptitle("E3: cách biểu diễn pixel và dạng phân phối của Naive Bayes (validation)", fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def plot_theta(log_theta, title, path):
    """θ_c = exp(log θ_c) của 10 lớp trên lưới 28×28; xám: pixel bị bỏ (NaN)."""
    theta = np.exp(nan_array(log_theta)).reshape(-1, *IMAGE_SHAPE)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#bdbdbd")
    for k, ax in enumerate(axes.flat):
        ax.imshow(np.ma.masked_invalid(theta[k]), cmap=cmap)
        ax.set_title(f"lớp {k}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def plot_all():
    summary = load_summary(KEY)
    fm = load_final_metrics(KEY)
    final = summary["final"]
    X_te, y_te = load_splits()["test"]
    clear_figures(KEY)
    fig = fig_dir(KEY)
    return [
        plot_e1(summary["e1_alpha"], final["config"]["alpha"], fig / "hp_e1_alpha.png"),
        plot_e3(summary["e3_input"], fig / "hp_e3_input.png"),
        plot_theta(final["log_theta"], "θ của Multinomial NB: phân phối mực của từng lớp (xám: pixel bị bỏ)",
                   fig / "interp_theta.png"),
        plot_confusion_splits(fm["splits"], fig / "eval_confusion.png", "Multinomial NB",
                              highlight_pairs=EDA_CLOSE_PAIRS),
        plot_confident_errors(X_te, y_te, final["confident_errors"], fig / "eval_confident_errors.png",
                              "Multinomial NB (test)"),
    ]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
