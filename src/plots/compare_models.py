"""Hình so sánh các mô hình, vẽ từ results/comparison/summary.json (do src/experiments/compare_models.py ghi).

Chạy: python -m src.plots.compare_models   (hoặc python -m src.plots để vẽ mọi mô hình)

    cmp_metrics.png           accuracy, F1 macro trên val và test
    cmp_time.png              thời gian huấn luyện và dự đoán (thang log, kèm số luồng)
    cmp_pair_errors.png       lỗi ở 5 cặp EDA 3.1 trên val (H3)
    cmp_per_class_f1.png      F1 theo lớp trên test
    cmp_block_importance.png  độ quan trọng theo khối 4×4 cạnh tổng Fisher của khối (H4)
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import EDA_CLOSE_PAIRS, MODEL_COLORS
from src.evaluation.results_io import clear_figures, fig_dir, load_summary
from src.plots.common import nan_array, savefig

KEY = "comparison"


def color(k, n=0):
    return MODEL_COLORS.get(k, f"C{n}")


def plot_metrics(s, path):
    keys, names, M = s["models"], s["names"], s["metrics"]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1_macro", "F1 macro")]:
        for off, split, hatch in [(-0.2, "val", ""), (0.2, "test", "//")]:
            vals = [M[k][split][metric] for k in keys]
            bars = ax.bar(x + off, vals, 0.4, color=[color(k) for k in keys], hatch=hatch, edgecolor="white", label=split)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        lo = min(M[k][sp][metric] for k in keys for sp in ("val", "test"))
        ax.set(xticks=x, xticklabels=[names[k] for k in keys], ylim=(max(0, lo - 0.05), 1.0), title=title)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    return savefig(fig, path)


def plot_time(s, path):
    keys, names, M, jobs = s["models"], s["names"], s["metrics"], s["n_jobs"]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, vals, title in [(axes[0], [M[k]["fit_time"] for k in keys], "Thời gian huấn luyện (s)"),
                            (axes[1], [M[k]["test"]["predict_time"] for k in keys],
                             "Thời gian dự đoán 10.000 ảnh test (s)")]:
        vals = [np.nan if val is None else val for val in vals]
        bars = ax.bar(x, vals, color=[color(k) for k in keys])
        for b, val, k in zip(bars, vals, keys):
            if np.isfinite(val):
                ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3g}\n({jobs[k]} luồng)" if jobs[k] else f"{val:.3g}",
                        ha="center", va="bottom", fontsize=7)
        ax.set(xticks=x, xticklabels=[names[k] for k in keys], yscale="log", title=title + ", thang log")
        ax.tick_params(axis="x", labelsize=8, rotation=15)
    plt.tight_layout()
    return savefig(fig, path)


def plot_pair_errors(s, path):
    keys, names, pe = s["models"], s["names"], s["pair_errors_val"]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(keys)
    for n, k in enumerate(keys):
        ax.bar(np.arange(5) + (n - (len(keys) - 1) / 2) * width, pe[k], width, color=color(k, n), label=names[k])
    ax.set(xticks=range(5), xticklabels=[f"{i}–{j}" for i, j in EDA_CLOSE_PAIRS], ylabel="số lỗi (val, hai chiều)",
           title="Lỗi ở 5 cặp có tâm gần nhau nhất (EDA 3.1): H3")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    return savefig(fig, path)


def plot_per_class_f1(s, path):
    keys, names = s["models"], s["names"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for n, k in enumerate(keys):
        ax.plot(range(10), s["per_class_f1_test"][k], "o-", color=color(k, n), label=names[k])
    ax.set(xticks=range(10), xlabel="chữ số", ylabel="F1 trên test", title="F1 theo lớp")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    return savefig(fig, path)


def plot_block_importance(s, path):
    """s["block"]: {"fisher": tổng Fisher theo khối, "models": {mô hình: {"importance", "rho"}}}."""
    F_block = nan_array(s["block"]["fisher"])
    side = int(np.sqrt(len(F_block)))
    panels = [("Fisher, tổng theo khối", F_block)] + [
        (f"{s['names'][k]}\nρ = {b['rho']:.2f}", nan_array(b["importance"])) for k, b in s["block"]["models"].items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
    for ax, (title, M) in zip(axes, panels):
        cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#9e9e9e")
        ax.imshow(np.ma.masked_invalid(M.reshape(side, side)), cmap=cmap)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Độ quan trọng theo khối 4×4 (hoán vị trên val), cùng một cách đo cho mọi mô hình", fontsize=10)
    plt.tight_layout()
    return savefig(fig, path)


def plot_all():
    s = load_summary(KEY)
    clear_figures(KEY)
    fig = fig_dir(KEY)
    paths = [plot_metrics(s, fig / "cmp_metrics.png"), plot_time(s, fig / "cmp_time.png"),
             plot_pair_errors(s, fig / "cmp_pair_errors.png"), plot_per_class_f1(s, fig / "cmp_per_class_f1.png")]
    if s.get("block"):
        paths.append(plot_block_importance(s, fig / "cmp_block_importance.png"))
    return paths


if __name__ == "__main__":
    for path in plot_all():
        print(path)
