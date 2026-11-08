"""Hình của CNN, vẽ từ results/cnn/summary.json và final_metrics.json.

Chạy: python -m src.plots.cnn   (hoặc python -m src.plots để vẽ mọi mô hình)

    hp_e1_lr.png                E1: loss và val accuracy theo epoch, mỗi learning rate một màu
    hp_e2_width.png             E2: val accuracy theo thời gian huấn luyện, mỗi độ rộng một điểm
    eval_learning_curve.png     E3: đường học của cấu hình cuối
    eval_confusion.png          ma trận nhầm lẫn train | val | test
    eval_confident_errors.png   20 ảnh test bị đoán sai tự tin nhất
"""

import matplotlib.pyplot as plt

from src.config import EDA_CLOSE_PAIRS
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics, load_summary
from src.plots.common import COLORS, plot_confident_errors, plot_confusion_splits, savefig

KEY = "cnn"


def plot_e1(e1, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c, r in zip(COLORS, e1["rows"]):
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], "--", color=c)
        axes[0].plot(ep, [h["val_loss"] for h in r["history"]], "o-", color=c, label=f"lr = {r['lr']:g}")
        axes[1].plot(ep, [h["val_acc"] for h in r["history"]], "o-", color=c, label=f"lr = {r['lr']:g}")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss (nét liền: val, nét đứt: train)")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Val accuracy")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"E1. Learning rate của Adam (chọn {e1['chosen']:g})")
    plt.tight_layout()
    return savefig(fig, path)


def plot_e2(e2, path):
    rows, tol = e2["rows"], e2["tol"]
    best = max(r["val_acc_last"] for r in rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, r in zip(COLORS, rows):
        ax.scatter(r["fit_s"], r["val_acc_last"], s=60, color=c, zorder=3)
        ax.annotate(f"{tuple(r['channels'])}\n{r['n_params']:,} tham số".replace(",", "."),
                    (r["fit_s"], r["val_acc_last"]), textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.axhspan(best - tol, best, color="#9e9e9e", alpha=0.2, label=f"trong {tol} của tốt nhất")
    ax.set(xlabel="thời gian huấn luyện (s)", ylabel="val accuracy",
           title=f"E2. Độ rộng CNN (chọn {tuple(e2['chosen'])})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return savefig(fig, path)


def plot_learning_curve(e3, path):
    hist = e3["history"]
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, name in [(axes[0], "loss", "cross-entropy"), (axes[1], "acc", "accuracy")]:
        ax.plot(ep, [h[f"train_{key}"] for h in hist], "o--", color=COLORS[0], label="train (TB trong epoch)")
        ax.plot(ep, [h[f"val_{key}"] for h in hist], "o-", color=COLORS[1], label="val")
        ax.axvline(e3["best_epoch"], color="#6b6b6b", ls=":", label=f"epoch giữ lại = {e3['best_epoch']}")
        ax.set(xlabel="epoch", ylabel=name)
        ax.legend()
    fig.suptitle("E3. Đường học của cấu hình cuối")
    plt.tight_layout()
    return savefig(fig, path)


def plot_all():
    summary = load_summary(KEY)
    fm = load_final_metrics(KEY)
    X_te, y_te = load_splits()["test"]
    clear_figures(KEY)
    fig = fig_dir(KEY)
    return [
        plot_e1(summary["E1"], fig / "hp_e1_lr.png"),
        plot_e2(summary["E2"], fig / "hp_e2_width.png"),
        plot_learning_curve(summary["E3"], fig / "eval_learning_curve.png"),
        plot_confusion_splits(fm["splits"], fig / "eval_confusion.png", "CNN", highlight_pairs=EDA_CLOSE_PAIRS),
        plot_confident_errors(X_te, y_te, summary["final"]["confident_errors"], fig / "eval_confident_errors.png",
                              "CNN (test)"),
    ]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
