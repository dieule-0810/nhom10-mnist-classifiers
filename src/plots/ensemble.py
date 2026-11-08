"""Hình của phần gộp mô hình, vẽ từ results/ensemble/summary.json và final_metrics.json.

Chạy: python -m src.plots.ensemble   (hoặc python -m src.plots để vẽ mọi mô hình)

    div_pairs.png              E1: "% cùng sai" trên val cho mọi cặp thành viên
    ens_seeds.png              E3: val accuracy theo số CNN khác seed, soft và hard voting
    ens_weights.png            E4: val accuracy theo trọng số của CNN ensemble khi trộn với mô hình cổ điển
    eval_confusion.png         ma trận nhầm lẫn train | val | test của phương án chọn trên val
    eval_residual_errors.png   20 ảnh test mà phương án đã chọn vẫn đoán sai tự tin nhất
"""

import matplotlib.pyplot as plt

from src.config import DISPLAY_NAMES, ENSEMBLE_KEY, MODEL_COLORS
from src.data.preprocess import load_splits
from src.evaluation.results_io import clear_figures, fig_dir, load_final_metrics, load_summary
from src.plots.common import nan_array, plot_confident_errors, plot_confusion_splits, savefig

KEY = ENSEMBLE_KEY


def plot_div_pairs(e1, path):
    """e1["matrix"]: % cùng sai giữa thành viên i và j (đường chéo NaN); e1["names"]: tên thành viên."""
    M, names = nan_array(e1["matrix"]), e1["names"]
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.0 * n + 1.5))
    im = ax.imshow(M, cmap="OrRd", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "black")
    ax.set(xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names,
           title="E1. % cùng sai trên val (thấp = khác nhau = đáng gộp)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return savefig(fig, path)


def plot_seeds(e3, path):
    rows = e3["rows"]
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(ks, [r["soft_min"] for r in rows], [r["soft_max"] for r in rows], color=MODEL_COLORS["cnn"],
                    alpha=0.15, label="soft: khoảng min–max mọi tổ hợp k mạng")
    ax.plot(ks, [r["soft_mean"] for r in rows], "o-", color=MODEL_COLORS["cnn"], label="soft voting (TB các tổ hợp)")
    ax.plot(ks, [r["hard_first"] for r in rows], "s--", color="#6b6b6b", label="hard voting (k seed đầu)")
    ax.set(xlabel="số mạng CNN (khác seed)", ylabel="val accuracy", xticks=ks, title="E3. Seed ensemble CNN")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return savefig(fig, path)


def plot_weights(e4, path):
    """e4["rows"]: mỗi tổ hợp cổ điển một dòng, accs = {trọng số của CNN ensemble: val accuracy}."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    weights = None
    for r in e4["rows"]:
        weights = [int(w) for w in r["accs"]]
        ax.plot(weights, list(r["accs"].values()), "o-", label=f"CNN-ens + {' + '.join(r['combo'])}")
    ax.axhline(e4["cnn_only"], color=MODEL_COLORS["cnn"], ls=":", label="chỉ CNN ensemble")
    ax.set(xlabel="trọng số w của CNN ensemble (mỗi mô hình cổ điển: 1)", ylabel="val accuracy", xticks=weights,
           title="E4. Weighted soft voting")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return savefig(fig, path)


def plot_all():
    summary = load_summary(KEY)
    fm = load_final_metrics(KEY)
    final = summary["E5"]["final"]
    X_te, y_te = load_splits()["test"]
    clear_figures(KEY)
    fig = fig_dir(KEY)
    paths = [plot_div_pairs(summary["E1"], fig / "div_pairs.png")]
    if summary.get("E3"):
        paths.append(plot_seeds(summary["E3"], fig / "ens_seeds.png"))
    if summary["E4"].get("rows"):
        paths.append(plot_weights(summary["E4"], fig / "ens_weights.png"))
    paths += [
        plot_confusion_splits(fm["splits"], fig / "eval_confusion.png", f"{DISPLAY_NAMES[KEY]}: {final['name']}"),
        plot_confident_errors(X_te, y_te, final["confident_errors"], fig / "eval_residual_errors.png",
                              f"{final['name']} (test): những ảnh ensemble vẫn sai"),
    ]
    return paths


if __name__ == "__main__":
    for path in plot_all():
        print(path)
