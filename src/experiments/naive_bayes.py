"""Thí nghiệm Naive Bayes: Multinomial NB trên pixel [0, 1], nối từ EDA (eda/notebooks/eda.ipynb) tới đánh giá cuối.

Chạy từ gốc repo:
    python -m src.experiments.naive_bayes            # đầy đủ, dưới 1 phút trên CPU laptop
    python -m src.experiments.naive_bayes --quick    # lưới nhỏ, chỉ để kiểm tra code chạy được

Kết quả ghi vào results/naive_bayes/ (quy ước chung: results/README.md):
    log.txt             toàn bộ bảng in ra màn hình, kể cả 3 ma trận nhầm lẫn
    summary.json        số liệu của các thí nghiệm E1-E4 (mốc E0 chỉ in vào log.txt)
    final_metrics.json  mô hình cuối, cấu trúc chung với LR, MLP, RF (đầu vào của compare_models)
    figures/            hp_*.png, eval_*.png, interp_*.png: do src/plots/naive_bayes.py vẽ từ summary.json và
                        final_metrics.json (gọi ở cuối script; vẽ lại riêng: python -m src.plots.naive_bayes)

Quy tắc: mỗi thí nghiệm chỉ đổi MỘT yếu tố, chọn theo VALIDATION. Tập test chỉ dùng MỘT lần ở E4.
NB không có thành phần ngẫu nhiên, nên chênh lệch accuracy chỉ là chênh lệch thật trên 10.000 ảnh val;
dưới ACC_TOL (10 ảnh) coi như bằng nhau và chọn phương án ít đặc trưng hơn (cùng tinh thần LR, RF).

    E0  Mốc thấp nhất: tâm gần nhất                                  (EDA 3.1)
    E1  Hệ số làm trơn α của Multinomial NB                          (EDA 1.1, 5.1: α > 0 bắt buộc)
    E2  Chọn pixel: giữ hết / bỏ dead / bỏ hiếm / crop 24×24, 20×20  (EDA 1.1, 1.3)
    E3  Đối chứng: Multinomial trên cường độ [0, 1] (EDA 5.1) so với nhị phân x > 0, thang [0, 255]
        (EDA 5.1: thang đo đổi mức tự tin) và Gaussian NB trên [0, 1] (EDA 5.1: giả định chuẩn sai)
    E4  Mô hình cuối: train / val / test, xác suất, lỗi theo cặp, θ và độ quan trọng
"""

import argparse
import json
import time

import numpy as np

from src.config import EDA_CLOSE_PAIRS, PERM_REPEATS, SEED
from src.data.features import FeaturePipeline, PixelMask, Quantizer, make_features
from src.data.preprocess import load_splits
from src.evaluation.analysis import (block_permutation_importance, confident_error_records, pair_errors,
                                     probabilistic_scores, share_in_pairs, top_confused_pairs)
from src.evaluation.metrics import evaluate, summary_table
from src.evaluation.results_io import (Logger, compact_eval, format_confusion, model_dir, rel_path,
                                       save_final_metrics, write_json)
from src.models.naive_bayes import GaussianNBNP, MultinomialNBNP
from src.models.nearest_centroid import NearestCentroidNP
from src.models.pipeline import Pipeline
from src.plots.naive_bayes import plot_all

KEY = "naive_bayes"
OUT_DIR = model_dir(KEY)

FULL = dict(alphas=[1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            masks=["none", "dead", "rare", "crop24", "crop20"],
            e3_alphas=[1e-3, 1e-2, 0.1, 1.0], e3_var_smoothing=[1e-9, 1e-3, 1e-2, 0.1, 0.3, 1.0], perm_repeats=PERM_REPEATS)
QUICK = dict(FULL, alphas=[1e-2, 1.0], masks=["none", "crop24"], e3_alphas=[1e-2], e3_var_smoothing=[1e-2],
             perm_repeats=1)

# Chênh val accuracy dưới mức này (10 ảnh / 10.000) coi như bằng nhau; khi đó chọn cấu hình ít đặc trưng hơn.
ACC_TOL = 0.001

class Scale:
    """Nhân đặc trưng với một hằng số (E3: thang [0, 255] thay cho [0, 1])."""

    def __init__(self, c):
        self.c = c

    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64) * self.c

    def fit_transform(self, X):
        return self.transform(X)

    def __repr__(self):
        return f"Scale({self.c:g})"


# ---------------------------------------------------------------------------
# Hàm dùng chung
# ---------------------------------------------------------------------------
def fit_val(splits, features, model):
    """Huấn luyện Pipeline trên train, đo trên val; trả về (pipeline, dict số liệu val)."""
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    pipe = Pipeline(features, model)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    proba = pipe.predict_proba(X_va)
    ps = probabilistic_scores(y_va, proba)
    row = {"val_acc": float((proba.argmax(axis=1) == y_va).mean()), "val_loss": ps["log_loss"],
           "overconfident_errors": ps["overconfident_errors"],
           "overconfident_share_of_errors": ps["overconfident_share_of_errors"],
           "fit_time": fit_time, "n_features": int(pipe.model.n_features_in_)}
    return pipe, row


def pick_parsimonious(rows):
    """rows: {lựa chọn: row}. Ít đặc trưng nhất trong các lựa chọn có val acc ≥ val acc lớn nhất − ACC_TOL."""
    best = max(r["val_acc"] for r in rows.values())
    near = {k: r for k, r in rows.items() if r["val_acc"] >= best - ACC_TOL}
    return min(near, key=lambda k: (near[k]["n_features"], -near[k]["val_acc"]))


# ---------------------------------------------------------------------------
# E0. Mốc thấp nhất
# ---------------------------------------------------------------------------
def run_e0(splits, log):
    log("\n=== E0. Tâm gần nhất (mốc) ===")
    nc = NearestCentroidNP()
    res = evaluate(nc, {k: splits[k] for k in ("train", "val")})
    log(f"val acc = {res['val']['accuracy']:.4f} | train acc = {res['train']['accuracy']:.4f}")
    return nc, res


# ---------------------------------------------------------------------------
# E1. α
# ---------------------------------------------------------------------------
def run_e1(splits, cfg, log):
    log("\n=== E1. Hệ số làm trơn α (Multinomial NB, pixel [0, 1], giữ cả 784 pixel) ===")
    log(f"{'α':>7} | {'val acc':>7} | {'val loss':>8} | {'sai mà p>0.99':>13} | {'fit (s)':>7}")
    rows = {}
    for a in cfg["alphas"]:
        _, row = fit_val(splits, make_features("none", "none"), MultinomialNBNP(alpha=a))
        rows[a] = row
        log(f"{a:7g} | {row['val_acc']:7.4f} | {row['val_loss']:8.4f} | "
            f"{row['overconfident_errors']:13.2%} | {row['fit_time']:7.2f}")
    accs = np.array([r["val_acc"] for r in rows.values()])
    # Luật chọn: val accuracy lớn nhất; bằng nhau thì val loss nhỏ hơn
    best = max(rows, key=lambda a: (rows[a]["val_acc"], -rows[a]["val_loss"]))
    log(f"Chọn α = {best:g} (val acc lớn nhất). Chênh val acc giữa mọi α: {accs.max() - accs.min():.4f} "
        f"({round((accs.max() - accs.min()) * len(splits['val'][1]))} ảnh)")
    return best, rows


# ---------------------------------------------------------------------------
# E2. Chọn pixel
# ---------------------------------------------------------------------------
def run_e2(splits, cfg, log, alpha):
    log(f"\n=== E2. Chọn pixel (α = {alpha:g}) ===")
    log("Bỏ pixel đổi số chiều d nên đổi mẫu số làm trơn α·d: mỗi mặt nạ là một mô hình khác, chọn trên val")
    log(f"{'mask':>7} | {'số đặc trưng':>12} | {'val acc':>7} | {'val loss':>8} | {'sai mà p>0.99':>13}")
    rows = {}
    for mask in cfg["masks"]:
        _, row = fit_val(splits, make_features(mask, "none"), MultinomialNBNP(alpha=alpha))
        rows[mask] = row
        log(f"{mask:>7} | {row['n_features']:12d} | {row['val_acc']:7.4f} | {row['val_loss']:8.4f} | "
            f"{row['overconfident_errors']:13.2%}")
    chosen = pick_parsimonious(rows)
    log(f"Chọn mask = {chosen} (ít đặc trưng nhất trong các mặt nạ có val acc cách tốt nhất ≤ {ACC_TOL})")
    return chosen, rows


# ---------------------------------------------------------------------------
# E3. Đối chứng cách biểu diễn đầu vào
# ---------------------------------------------------------------------------
def run_e3(splits, cfg, log, alpha, mask):
    log(f"\n=== E3. Đối chứng cách biểu diễn và mô hình (mask = {mask}; mỗi dòng chọn tham số tốt nhất trên val: "
        f"α ∈ {cfg['e3_alphas']}, var_smoothing ∈ {cfg['e3_var_smoothing']}) ===")
    log("EDA 5.1 chọn Multinomial trên cường độ [0, 1] ('lượng mực'); các dòng khác chỉ để kiểm chứng lựa chọn đó")
    masked = lambda: [PixelMask(mask)] if mask != "none" else []
    multinomial = [("α", a, MultinomialNBNP) for a in cfg["e3_alphas"]]
    variants = {                                              # tên: (đặc trưng, lưới [(tên tham số, giá trị, lớp)])
        "Multinomial, x ∈ [0, 1]": (lambda: FeaturePipeline(masked()), [("α", alpha, MultinomialNBNP)]),
        "Multinomial, x > 0": (lambda: FeaturePipeline(masked() + [Quantizer(2)]), multinomial),
        "Multinomial, x ∈ [0, 255]": (lambda: FeaturePipeline(masked() + [Scale(255)]), multinomial),
        "Gaussian, x ∈ [0, 1]": (lambda: FeaturePipeline(masked()),
                                 [("var_smoothing", v, GaussianNBNP) for v in cfg["e3_var_smoothing"]]),
    }
    log(f"{'biểu diễn':>26} | {'tham số':>19} | {'val acc':>7} | {'val loss':>8} | {'lỗi có p>0.99':>13}")
    rows = {}
    for name, (feats, grid) in variants.items():
        best, by_param = None, {}
        for pname, pval, cls in grid:
            _, row = fit_val(splits, feats(), cls(**{"alpha" if pname == "α" else pname: pval}))
            by_param[f"{pval:g}"] = row["val_acc"]
            if best is None or (row["val_acc"], -row["val_loss"]) > (best["val_acc"], -best["val_loss"]):
                best = dict(row, param=pname, param_value=pval)
        rows[name] = dict(best, val_acc_by_param=by_param)
        log(f"{name:>26} | {pname + ' = ' + format(best['param_value'], 'g'):>19} | {best['val_acc']:7.4f} | "
            f"{best['val_loss']:8.4f} | {best['overconfident_share_of_errors']:13.1%}")
    log("val acc theo tham số: " + "; ".join(f"{n}: " + ", ".join(f"{k} → {v:.4f}" for k, v in r["val_acc_by_param"].items())
                                          for n, r in rows.items() if len(r["val_acc_by_param"]) > 1))
    return rows


# ---------------------------------------------------------------------------
# E4. Mô hình cuối
# ---------------------------------------------------------------------------
def run_e4(splits, cfg, log, final_cfg, nc, quick):
    log("\n=== E4. Mô hình cuối (test dùng một lần) ===")
    log("Cấu hình:", json.dumps(final_cfg, ensure_ascii=False))
    X_va, y_va = splits["val"]
    pipe = Pipeline(make_features(final_cfg["mask"], "none"), MultinomialNBNP(alpha=final_cfg["alpha"]),
                    name="Multinomial NB")
    res = evaluate(pipe, splits)                              # fit trên train; train, val, test - test chỉ ở đây
    res["model"] = "Multinomial NB"
    log(f"fit {res['fit_time']:.2f}s (gồm chọn pixel)")
    nc_full = evaluate(nc, splits)
    nc_full["model"] = "Tâm gần nhất"
    for split in ("train", "val", "test"):
        log(f"\n[{split}]")
        log(summary_table([nc_full, res], split))

    # --- xác suất ---
    log("\nĐộ đo xác suất (so với Logistic Regression):")
    prob = {}
    for split in ("train", "val", "test"):
        X, y = splits[split]
        prob[split] = probabilistic_scores(y, pipe.predict_proba(X))
        p = prob[split]
        log(f"  {split:>5}: log-loss {p['log_loss']:.4f} | sai mà p>0.99: "
            f"{p['overconfident_errors']:.2%} mẫu ({p['overconfident_share_of_errors']:.1%} số lỗi) | "
            f"độ tự tin TB {p['mean_confidence']:.3f}")

    # --- lỗi theo cặp ---
    log("\nLỗi theo cặp lớp (cộng hai chiều). Dấu * đánh dấu 5 cặp EDA 3.1 dự báo:")
    pairs_out = {}
    for split in ("val", "test"):
        for name, r in [("NB", res), ("Tâm gần nhất", nc_full)]:
            cm = r[split]["confusion_matrix"]
            top = top_confused_pairs(cm, 8)
            share = share_in_pairs(cm)
            pairs_out[f"{name}/{split}"] = {"top": [[[i, j], n] for i, j, n in top], "share_eda_pairs": share,
                                            "eda_pairs": {f"{i}-{j}": n for (i, j), n in pair_errors(cm, EDA_CLOSE_PAIRS).items()}}
            txt = ", ".join(f"{i}–{j}{'*' if (i, j) in EDA_CLOSE_PAIRS else ''}: {n}" for i, j, n in top)
            log(f"  {name:>12} {split:>4}: {txt}")
            log(f"  {'':>12} {'':>4}  5 cặp EDA chiếm {share:.1%} tổng lỗi (nếu rải đều: {5 / 45:.1%})")

    log("\nMa trận nhầm lẫn của Multinomial NB (hàng: nhãn thật, cột: dự đoán; cột cuối: recall của lớp):")
    for split in ("train", "val", "test"):
        log(format_confusion(res[split]["confusion_matrix"], f"\n[{split}]"))

    # --- θ và độ quan trọng; so với Fisher (H4) làm một lần cho mọi mô hình ở compare_models ---
    log_theta = pipe.pixel_weights().reshape(10, -1)          # log θ, (10, 784), NaN ở pixel bị bỏ
    t0 = time.perf_counter()
    blocks = block_permutation_importance(pipe, X_va, y_va, n_repeats=cfg["perm_repeats"], seed=SEED)
    log(f"\nSố đo chung cho mọi mô hình: hoán vị khối 4×4 trên val, khối quan trọng nhất làm giảm acc "
        f"{blocks.max():.4f} ({time.perf_counter() - t0:.0f}s)")

    X_te, y_te = splits["test"]
    errors = confident_error_records(y_te, pipe.predict_proba(X_te))

    save_final_metrics(
        KEY, "Multinomial Naive Bayes", res, final_cfg, probabilistic=prob,
        block_importance=blocks, n_jobs=1, quick=quick)

    return {"config": final_cfg, "fit_time": res["fit_time"], "naive_bayes": compact_eval(res),
            "nearest_centroid": compact_eval(nc_full), "probabilistic": prob, "pairs": pairs_out,
            "log_theta": log_theta, "confident_errors": errors}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="lưới nhỏ, chỉ để kiểm tra code")
    quick = ap.parse_args().quick

    cfg = QUICK if quick else FULL
    t_start = time.perf_counter()
    log = Logger(OUT_DIR / "log.txt")
    splits = load_splits()
    log(f"train {splits['train'][0].shape} | val {splits['val'][0].shape} | test {splits['test'][0].shape}")

    nc, _ = run_e0(splits, log)
    alpha, e1 = run_e1(splits, cfg, log)
    mask, e2 = run_e2(splits, cfg, log, alpha)
    e3 = run_e3(splits, cfg, log, alpha, mask)
    final_cfg = dict(variant="multinomial", alpha=alpha, mask=mask, features="pixel [0, 1], không chuẩn hoá, không PCA")
    e4 = run_e4(splits, cfg, log, final_cfg, nc, quick)

    log(f"\nTổng thời gian: {time.perf_counter() - t_start:.0f}s")
    summary = {"quick": quick, "acc_tol": ACC_TOL, "e1_alpha": {str(k): v for k, v in e1.items()}, "e2_mask": e2,
               "e3_input": e3, "final": e4}
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'summary.json')}, log.txt và figures/")


if __name__ == "__main__":
    main()
