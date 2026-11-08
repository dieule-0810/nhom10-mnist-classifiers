"""Hình của Logistic Regression, vẽ từ results/logistic_regression/summary.json và final_metrics.json.

Chạy: python -m src.plots.logistic_regression   (hoặc python -m src.plots để vẽ mọi mô hình)

    hp_e2a_lr_momentum.png       E2a: val accuracy theo learning rate × momentum
    hp_e2c_batch_size.png        E2c: val loss, val accuracy, giây / epoch theo kích thước batch
    hp_e2b_convergence.png       E2b: J(W) và val accuracy theo epoch, full-batch so với mini-batch
    hp_e3_l2.png                 E3: mất mát, val accuracy, độ lốm đốm theo λ
    hp_e3_weights_by_lambda.png  E3: ảnh trọng số theo λ
    interp_weights.png           ảnh trọng số của mô hình cuối và của tâm gần nhất
    eval_learning_curve.png      đường học của mô hình cuối
    eval_confusion.png           ma trận nhầm lẫn train | val | test
    eval_confident_errors.png    20 ảnh test bị đoán sai tự tin nhất
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import EDA_CLOSE_PAIRS, IMAGE_SHAPE
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics, load_summary
from src.plots.common import nan_array, plot_confident_errors, plot_confusion_splits, savefig

KEY = "logistic_regression"


def plot_e2a(e2, path):
    """Lưới val accuracy (hàng: momentum, cột: η); ô NaN là cấu hình phân kỳ."""
    etas, moms = e2["etas"], e2["momenta"]
    acc = nan_array(e2["grid_acc"])
    fig, ax = plt.subplots(figsize=(1.6 * len(etas) + 2, 0.9 * len(moms) + 1.5))
    ax.imshow(np.nan_to_num(acc, nan=np.nanmin(acc)), cmap="viridis")
    for a in range(len(moms)):
        for b in range(len(etas)):
            txt = "phân kỳ" if np.isnan(acc[a, b]) else f"{acc[a, b]:.4f}"
            txt += f"\nη_eff={etas[b] / (1 - moms[a]):.3g}"
            ax.text(b, a, txt, ha="center", va="center", fontsize=8,
                    color="white" if np.isnan(acc[a, b]) or acc[a, b] < np.nanmedian(acc) else "black")
    ax.set_xticks(range(len(etas))); ax.set_xticklabels([f"η={e}" for e in etas])
    ax.set_yticks(range(len(moms))); ax.set_yticklabels([f"α={a}" for a in moms])
    ax.set_xlabel("learning rate"); ax.set_ylabel("momentum")
    ax.set_title(f"Val accuracy sau {e2['epochs']} epoch")
    plt.tight_layout()
    return savefig(fig, path)


def plot_e2c(e2c, path):
    """rows: mỗi batch một dòng với η tốt nhất của nó; batch_size None là full-batch."""
    rows, chosen, n_train, tol = e2c["rows"], e2c["chosen"], e2c["n_train"], e2c["tol"]
    best = min(rows, key=lambda r: r["val_loss"])
    mini = [r for r in rows if r["batch_size"] is not None]
    full = [r for r in rows if r["batch_size"] is None]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, key, label in [(axes[0], "val_loss", "val loss"), (axes[1], "val_acc", "val accuracy"),
                           (axes[2], "sec_per_epoch", "giây / epoch")]:
        ax.plot([r["batch_size"] for r in mini], [r[key] for r in mini], "o-", label="mini-batch (η tốt nhất)")
        if full and key == "sec_per_epoch":
            ax.axhline(full[0][key], ls="--", c="C3", label="full-batch")
        elif full:
            # full-batch kém xa (val loss ~0,64); vẽ cùng trục sẽ ép các điểm mini-batch thành một đường phẳng
            ax.set_title(f"full-batch (η = {full[0]['eta']:g}): {full[0][key]:.4f}", fontsize=9, color="C3")
        ax.axvline(chosen["batch_size"] or n_train, ls=":", c="gray", lw=0.8)
        ax.set(xscale="log", xlabel="kích thước batch (thang log)", ylabel=label)
    axes[0].axhspan(best["val_loss"], best["val_loss"] + tol, color="#f2c14e", alpha=0.3,
                    label=f"cách tốt nhất < {tol}")
    axes[0].legend(fontsize=8, frameon=False)
    for r in mini:
        axes[1].annotate(f"η={r['eta']:g}", (r["batch_size"], r["val_acc"]), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=7)
    fig.suptitle(f"E2c: kích thước batch sau {e2c['epochs']} epoch (α = {e2c['momentum']}; mỗi batch dùng "
                 f"η tốt nhất của nó)", fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def plot_e2b(curves, path):
    """curves: {tên cấu hình: history} với epoch, train_loss, val_acc."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, h in curves.items():
        line, = axes[0].plot(h["epoch"], h["train_loss"], lw=2, label=name)
        axes[1].plot(h["epoch"], h["val_acc"], lw=2, color=line.get_color())
    axes[0].set(xscale="log", xlabel="epoch (thang log)", ylabel="J(W) trên train", title="Hàm mất mát")
    axes[1].set(xscale="log", xlabel="epoch (thang log)", ylabel="accuracy", title="Val accuracy")
    axes[0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return savefig(fig, path)


def plot_e3(rows, path):
    """rows: {λ (chuỗi): số liệu val}."""
    lams = list(rows)
    xs = np.arange(len(lams))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(xs, [rows[l]["train_ce"] for l in lams], "o-", label="train")
    axes[0].plot(xs, [rows[l]["val_loss"] for l in lams], "o--", label="val")
    axes[0].set(ylabel="cross-entropy", title="Mất mát theo λ"); axes[0].legend(frameon=False)
    axes[1].plot(xs, [rows[l]["val_acc"] for l in lams], "o-")
    axes[1].set(ylabel="val accuracy", title="Val accuracy theo λ")
    axes[2].plot(xs, [rows[l]["roughness"] for l in lams], "o-")
    axes[2].set(ylabel="|Δw kề| / |w|", title="Độ lốm đốm của ảnh trọng số")
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels([f"{float(l):g}" for l in lams]); ax.set_xlabel("λ")
    plt.tight_layout()
    return savefig(fig, path)


def plot_weight_rows(rows, title, path):
    """rows: list (nhãn hàng, ảnh trọng số (10, 784)), NaN ở pixel bị bỏ."""
    fig, axes = plt.subplots(len(rows), 10, figsize=(13, 1.45 * len(rows) + 0.4), squeeze=False)
    for r, (label, W) in enumerate(rows):
        W = nan_array(W).reshape(-1, *IMAGE_SHAPE)
        W = W - np.nanmean(W, axis=0, keepdims=True)          # bỏ phần chung giữa các lớp (softmax bất biến)
        v = np.nanmax(np.abs(W))
        for k in range(10):
            ax = axes[r, k]
            cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#bdbdbd")
            ax.imshow(np.ma.masked_invalid(W[k]), cmap=cmap, vmin=-v, vmax=v)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(str(k), fontsize=10)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return savefig(fig, path)


def plot_learning_curve(h, best_epoch, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(h["epoch"], h["train_loss"], "-", label="train (J, có L2)")
    axes[0].plot(h["epoch"], h["val_loss"], "--", label="val (cross-entropy)")
    axes[1].plot(h["epoch"], h["train_acc"], "-", label="train")
    axes[1].plot(h["epoch"], h["val_acc"], "--", label="val")
    for ax in axes:
        ax.axvline(best_epoch, c="gray", lw=0.8, ls=":")
        ax.set_xlabel("epoch"); ax.legend(frameon=False, fontsize=8)
    axes[0].set(ylabel="mất mát", title="Mất mát"); axes[1].set(ylabel="accuracy", title="Accuracy")
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
        plot_e2a(summary["e2"], fig / "hp_e2a_lr_momentum.png"),
        plot_e2c(summary["e2c_batch"], fig / "hp_e2c_batch_size.png"),
        plot_e2b(summary["e2b_convergence"], fig / "hp_e2b_convergence.png"),
        plot_e3(summary["e3_l2"], fig / "hp_e3_l2.png"),
        plot_weight_rows([(f"λ={float(l):g}", W) for l, W in summary["e3_weights"].items()],
                         "Ảnh trọng số theo λ (đỏ: tăng điểm của lớp, xanh: giảm; xám: pixel bị bỏ)",
                         fig / "hp_e3_weights_by_lambda.png"),
        plot_weight_rows([("LR", final["weights"]), ("Tâm gần nhất\n(= ảnh TB)", final["nearest_centroid_weights"])],
                         "Ảnh trọng số theo pixel gốc (đã trừ phần chung giữa các lớp)", fig / "interp_weights.png"),
        plot_learning_curve(final["history"], final["best_epoch"], fig / "eval_learning_curve.png"),
        plot_confusion_splits(fm["splits"], fig / "eval_confusion.png", "Logistic Regression",
                              highlight_pairs=EDA_CLOSE_PAIRS),
        plot_confident_errors(X_te, y_te, final["confident_errors"], fig / "eval_confident_errors.png",
                              "Logistic Regression (test)"),
    ]


if __name__ == "__main__":
    for path in plot_all():
        print(path)
