"""Hình của Random Forest, vẽ từ results/random_forest/summary.json và final_metrics.json.

Chạy: python -m src.plots.random_forest   (hoặc python -m src.plots để vẽ mọi mô hình)

    hp_e1_n_bins.png            E1: val accuracy và thời gian theo số bin (trung bình ± độ lệch chuẩn theo seed)
    hp_e2_masks.png             E2: val accuracy, thời gian theo mặt nạ pixel; bản đồ pixel dead / gần như chết
    hp_e3_n_trees.png           E3: OOB và val accuracy theo số cây, soft / hard voting
    hp_e4_max_features.png      E4: val accuracy, sức mạnh, tương quan giữa các cây theo m
    hp_e5_leaf_depth.png        E5: train / val accuracy theo min_samples_leaf và max_depth
    interp_importance.png       tần suất có mực, MDI, hoán vị khối 4×4
    eval_confusion.png          ma trận nhầm lẫn train | val | test
    eval_confident_errors.png   20 ảnh test bị đoán sai tự tin nhất
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from src.config import EDA_CLOSE_PAIRS, IMAGE_SHAPE
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics, load_summary
from src.plots.common import (COLORS, nan_array, plot_confident_errors, plot_confusion_splits, plot_importance_maps,
                              savefig)

KEY = "random_forest"


def plot_e1(e1, path):
    """e1["stats"]: {số bin: {độ đo: [trung bình, độ lệch chuẩn]}}."""
    stats = e1["stats"]
    xs = [int(g) for g in stats]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, m, lab in [(axes[0], "val_acc", "val accuracy"), (axes[1], "fit_s", "thời gian huấn luyện (s)")]:
        ax.errorbar(xs, [stats[g][m][0] for g in stats], yerr=[stats[g][m][1] for g in stats], fmt="o-",
                    color=COLORS[0], capsize=3)
        ax.set(xscale="log", xlabel="số bin (thang log)", ylabel=lab)
        ax.set_xticks(xs); ax.set_xticklabels(xs)
    axes[0].set_title(f"Validation, {e1['n_estimators']} cây, trung bình {e1['n_seeds']} seed", fontsize=10)
    axes[1].set_title("Thời gian huấn luyện rừng", fontsize=10)
    return savefig(fig, path)


def plot_e2(e2, path):
    """Hai ô đầu: mỗi mặt nạ một cột; ô cuối: pixel_kinds (0 có mực, 1 gần như chết, 2 dead) và khung crop."""
    stats = e2["stats"]
    names = list(stats)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, m, lab in [(axes[0], "val_acc", "val accuracy"), (axes[1], "fit_s", "thời gian huấn luyện (s)")]:
        ax.bar(range(len(names)), [stats[g][m][0] for g in names], yerr=[stats[g][m][1] for g in names],
               color=[COLORS[i % len(COLORS)] for i in range(len(names))], capsize=3)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8); ax.set_ylabel(lab)
    lo = min(stats[g]["val_acc"][0] for g in names)
    hi = max(stats[g]["val_acc"][0] + stats[g]["val_acc"][1] for g in names)
    axes[0].set_ylim(lo - 0.01, hi + 0.005)
    axes[2].imshow(np.asarray(e2["pixel_kinds"]).reshape(IMAGE_SHAPE),
                   cmap=ListedColormap(["#e9edf3", "#e8833a", "#3b3b3b"]), vmin=0, vmax=2)
    for a, b, c in [(1.5, 25.5, COLORS[0]), (3.5, 23.5, COLORS[2])]:  # khung crop24 và crop20
        axes[2].add_patch(Rectangle((a, a), b - a, b - a, fill=False, ec=c, lw=1.5))
    axes[2].set_title("Đen: dead | cam: gần như chết\nkhung xanh dương: crop24 | khung xanh lá: crop20", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    return savefig(fig, path)


def plot_e3(e3, path):
    """e3["curves"]: {đường: mảng (số seed, số điểm)}; OOB có một điểm cho mỗi mốc trong e3["marks"]."""
    C = {k: np.asarray(v, dtype=float) for k, v in e3["curves"].items()}
    marks, t_max, t_sat = e3["marks"], e3["t_max"], e3["t_sat"]
    n_seeds = len(C["val_soft"])
    T = np.arange(1, t_max + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax in axes:
        for k, (key, lab) in enumerate([("oob_soft", "OOB (soft)"), ("val_soft", "validation (soft)"),
                                        ("val_hard", "validation (hard)")]):
            xs = marks if key == "oob_soft" else T
            m = C[key].mean(0)
            sd = C[key].std(0, ddof=1) if n_seeds > 1 else np.zeros(len(xs))
            ax.plot(xs, m, "o-" if key == "oob_soft" else "-", ms=4, color=COLORS[k], lw=1.6, label=lab)
            ax.fill_between(xs, m - sd, m + sd, color=COLORS[k], alpha=0.2)
    axes[0].set(xscale="log", xlabel="số cây T (thang log)", ylabel="accuracy", title="Toàn bộ khoảng T")
    axes[1].set(xlabel="số cây T", ylabel="accuracy", title="Phóng to vùng bão hoà")
    axes[1].set_xlim(10, t_max)
    if t_max > 10:
        oob_big = C["oob_soft"][:, [i for i, t in enumerate(marks) if t >= 10]]
        axes[1].set_ylim(min(oob_big.min(), C["val_hard"][:, 9:].min()) - 0.002, None)
    axes[1].axvline(t_sat, ls="--", c="gray", lw=0.8)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Accuracy theo số cây (trung bình ± độ lệch chuẩn, {n_seeds} seed; OOB tính ở các mốc)", y=1.03)
    return savefig(fig, path)


def plot_e4(e4, path):
    rows = e4["rows"]
    ms = [r["m"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, key, lab in [(axes[0], "val_acc", f"val accuracy ({e4['n_trees']} cây)"),
                         (axes[1], "strength", "accuracy trung bình của 1 cây"),
                         (axes[2], "correlation", "tương quan lỗi giữa hai cây")]:
        ax.plot(ms, [r[key] for r in rows], "o-", color=COLORS[0])
        ax.set(xscale="log", xlabel="m (thang log)", ylabel=lab)
        ax.axvline(int(np.sqrt(rows[0]["n_features"])), ls="--", c="gray", lw=0.8)
    fig.suptitle("Đánh đổi sức mạnh – tương quan khi đổi m (đường đứt: m = √d)")
    return savefig(fig, path)


def plot_e5(e5, path):
    rows = e5["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, sub, xlab in [(axes[0], [r for r in rows if r["vary"].startswith("min")], "min_samples_leaf"),
                          (axes[1], [r for r in rows if r["vary"].startswith("max")], "max_depth (None = không giới hạn)")]:
        labels = [r["vary"].split("=")[1] for r in sub]
        ax.plot(range(len(sub)), [r["train_acc"] for r in sub], "o-", color=COLORS[1], label="train")
        ax.plot(range(len(sub)), [r["val_acc"] for r in sub], "o-", color=COLORS[0], label="validation")
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(labels)
        ax.set(xlabel=xlab, ylabel="accuracy")
        ax.legend(frameon=False)
    fig.suptitle(f"Giới hạn kích thước cây ({e5['n_estimators']} cây)")
    return savefig(fig, path)


def plot_importance(maps, path):
    """maps: mảng 784 của tần suất có mực, MDI (NaN ở pixel không bao giờ được dùng để chia), hoán vị khối."""
    return plot_importance_maps([("(a) Tần suất có mực (EDA 1.1)", nan_array(maps["ink_freq"])),
                                 ("(b) MDI của RF", nan_array(maps["mdi"])),
                                 ("(c) Hoán vị khối 4×4 trên val", nan_array(maps["block"]))], path)


def plot_all():
    summary = load_summary(KEY)
    fm = load_final_metrics(KEY)
    final = summary["final"]
    X_te, y_te = load_splits()["test"]
    clear_figures(KEY)
    fig = fig_dir(KEY)
    return [
        plot_e1(summary["E1"], fig / "hp_e1_n_bins.png"),
        plot_e2(summary["E2"], fig / "hp_e2_masks.png"),
        plot_e3(summary["E3"], fig / "hp_e3_n_trees.png"),
        plot_e4(summary["E4"], fig / "hp_e4_max_features.png"),
        plot_e5(summary["E5"], fig / "hp_e5_leaf_depth.png"),
        plot_importance(final["importance_maps"], fig / "interp_importance.png"),
        plot_confusion_splits(fm["splits"], fig / "eval_confusion.png", "Random Forest", highlight_pairs=EDA_CLOSE_PAIRS),
        plot_confident_errors(X_te, y_te, final["confident_errors"], fig / "eval_confident_errors.png",
                              "Random Forest (test)"),
    ]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
