"""Hình của MLP, vẽ từ results/mlp/summary.json và final_metrics.json.

Chạy: python -m src.plots.mlp   (hoặc python -m src.plots để vẽ mọi mô hình)

    hp_e1_lr_momentum.png       E1: val accuracy theo learning rate γ × momentum α
    hp_e2_batch_size.png        E2: val accuracy và giây / epoch theo kích thước batch
    hp_e3_architecture.png      E3: train / val accuracy theo kiến trúc
    hp_e4_activation.png        E4: chuẩn gradient từng lớp lúc khởi tạo theo hàm kích hoạt
    hp_e5_init.png              E5: std đầu ra ReLU qua 10 lớp theo cách khởi tạo
    hp_e6_regularization.png    E6: val loss, val accuracy theo epoch với L2 / dropout
    eval_learning_curve.png     đường học của mô hình cuối
    interp_w1.png               phần trọng số lớp ẩn 1 học từ dữ liệu, 16 nơ-ron mạnh nhất
    eval_confusion.png          ma trận nhầm lẫn train | val | test
    eval_confident_errors.png   20 ảnh test bị đoán sai tự tin nhất
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import EDA_CLOSE_PAIRS, IMAGE_SHAPE
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics, load_summary
from src.plots.common import nan_array, plot_confident_errors, plot_confusion_splits, savefig

KEY = "mlp"


def plot_e1(e1, path):
    lrs, momenta = e1["lrs"], e1["momenta"]
    acc = nan_array(e1["val_acc"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.imshow(np.nan_to_num(acc, nan=np.nanmin(acc)), cmap="viridis")
    for i in range(len(momenta)):
        for j in range(len(lrs)):
            txt = "phân kỳ" if np.isnan(acc[i, j]) else f"{acc[i, j]:.4f}"
            ax.text(j, i, txt + f"\nγ_eff={lrs[j] / (1 - momenta[i]):.3g}", ha="center", va="center", fontsize=8,
                    color="white" if np.isnan(acc[i, j]) or acc[i, j] < 0.8 else "black")
    ax.set(xticks=range(len(lrs)), xticklabels=[f"γ={l}" for l in lrs],
           yticks=range(len(momenta)), yticklabels=[f"α={a}" for a in momenta],
           title=f"E1: val accuracy sau {e1['epochs']} epoch")
    plt.tight_layout()
    return savefig(fig, path)


def plot_e2(e2, path):
    rows = e2["rows"]
    bs = [r["batch_size"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(bs, [r["val_acc"] for r in rows], "o-")
    axes[0].set(xscale="log", xlabel="batch (log)", ylabel="val accuracy", title="Val accuracy")
    axes[1].plot(bs, [r["s_per_epoch"] for r in rows], "o-", color="C1")
    axes[1].set(xscale="log", xlabel="batch (log)", ylabel="giây / epoch", title="Thời gian")
    fig.suptitle(f"E2: kích thước batch ({e2['epochs']} epoch, γ = {e2['lr']}, α = {e2['momentum']})")
    plt.tight_layout()
    return savefig(fig, path)


def plot_e3(e3, path):
    rows = e3["rows"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["train_acc"] for r in rows], 0.4, label="train")
    ax.bar(x + 0.2, [r["val_acc"] for r in rows], 0.4, label="val")
    ax.set(xticks=x, xticklabels=[r["name"] for r in rows], ylim=(0.9, 1.0), ylabel="accuracy",
           title=f"E3: độ rộng và độ sâu ({e3['epochs']} epoch)")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    plt.tight_layout()
    return savefig(fig, path)


def plot_e4(e4, path):
    grad_norms, act_acc = e4["grad_norms"], e4["val_acc"]
    n_layers = len(next(iter(grad_norms.values())))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.27
    for k, act in enumerate(grad_norms):
        ax.bar(np.arange(1, n_layers + 1) + (k - 1) * w, grad_norms[act], width=w,
               label=f"{act} (val acc {act_acc[act]:.3f})")
    ax.set(yscale="log", xlabel="lớp l", ylabel="‖∂J/∂W^(l)‖ (log)",
           title="E4: chuẩn gradient từng lớp lúc khởi tạo, mạng 6 lớp ẩn")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return savefig(fig, path)


def plot_e5(e5, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for init, stds in e5["relu_std"].items():
        ax.plot(range(1, len(stds) + 1), stds, "o-", label=init)
    ax.set(yscale="log", xlabel="lớp ẩn", ylabel="std đầu ra ReLU (log)",
           title="E5b: tín hiệu qua 10 lớp ReLU lúc khởi tạo")
    ax.legend()
    plt.tight_layout()
    return savefig(fig, path)


def plot_e6(e6, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, h in e6["histories"].items():
        ep = np.arange(1, len(h["val_loss"]) + 1)
        axes[0].plot(ep, h["val_loss"], label=name)
        axes[1].plot(ep, h["val_acc"], label=name)
    axes[0].set(xlabel="epoch", ylabel="val cross-entropy", title="Val loss")
    axes[1].set(xlabel="epoch", ylabel="val accuracy", title="Val accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"E6: điều chuẩn ({e6['epochs']} epoch)")
    plt.tight_layout()
    return savefig(fig, path)


def plot_learning_curve(h, best_epoch, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ep = np.arange(1, len(h["val_loss"]) + 1)
    axes[0].plot(ep, h["train_loss"], "-", label="train")
    axes[0].plot(ep, h["val_loss"], "--", label="val")
    axes[1].plot(ep, h["train_acc"], "-", label="train")
    axes[1].plot(ep, h["val_acc"], "--", label="val")
    for ax in axes:
        ax.axvline(best_epoch, color="gray", ls=":", lw=1)
        ax.legend()
        ax.set_xlabel("epoch")
    axes[0].set(ylabel="cross-entropy", title="Mất mát")
    axes[1].set(ylabel="accuracy", title="Accuracy")
    fig.suptitle(f"Mô hình cuối: đường học (đường chấm: epoch tốt nhất = {best_epoch})")
    plt.tight_layout()
    return savefig(fig, path)


def plot_w1(w1_top, path):
    """w1_top: {"idx": chỉ số nơ-ron, "R": phần học từ dữ liệu (16, 784), "n_hidden": số nơ-ron lớp 1}."""
    R = np.asarray(w1_top["R"], dtype=float)
    fig, axes = plt.subplots(2, 8, figsize=(13, 4))
    for j, r, ax in zip(w1_top["idx"], R, axes.ravel()):
        v = np.abs(r).max()
        ax.imshow(r.reshape(IMAGE_SHAPE), cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"nơ-ron {j}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Phần trọng số lớp ẩn 1 học từ dữ liệu: 16 nơ-ron mạnh nhất (trên {w1_top['n_hidden']})")
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
        plot_e1(summary["E1"], fig / "hp_e1_lr_momentum.png"),
        plot_e2(summary["E2"], fig / "hp_e2_batch_size.png"),
        plot_e3(summary["E3"], fig / "hp_e3_architecture.png"),
        plot_e4(summary["E4"], fig / "hp_e4_activation.png"),
        plot_e5(summary["E5"], fig / "hp_e5_init.png"),
        plot_e6(summary["E6"], fig / "hp_e6_regularization.png"),
        plot_learning_curve(summary["E7"]["history"], summary["E7"]["best_epoch"], fig / "eval_learning_curve.png"),
        plot_w1(final["w1_top"], fig / "interp_w1.png"),
        plot_confusion_splits(fm["splits"], fig / "eval_confusion.png", "MLP", highlight_pairs=EDA_CLOSE_PAIRS),
        plot_confident_errors(X_te, y_te, final["confident_errors"], fig / "eval_confident_errors.png", "MLP (test)"),
    ]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
