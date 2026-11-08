"""Thí nghiệm Logistic Regression: tiền xử lý -> huấn luyện -> đánh giá, kiểm chứng giả thuyết từ EDA.

Chạy từ gốc repo:
    python -m src.experiments.logistic_regression            # đầy đủ, khoảng 4 phút trên CPU laptop
    python -m src.experiments.logistic_regression --quick    # ít epoch, chỉ để kiểm tra code chạy được

Kết quả ghi vào results/logistic_regression/ (quy ước chung: results/README.md):
    log.txt             toàn bộ bảng in ra màn hình, kể cả 3 ma trận nhầm lẫn
    summary.json        số liệu của các thí nghiệm E1-E5 (mốc E0 chỉ in vào log.txt)
    final_metrics.json  mô hình cuối, cấu trúc chung với NB, MLP, RF (đầu vào của compare_models)
    figures/            hp_*.png, eval_*.png, interp_*.png: do src/plots/logistic_regression.py vẽ từ summary.json
                        và final_metrics.json (gọi ở cuối script; vẽ lại riêng: python -m src.plots.logistic_regression)
Baseline results/nearest_centroid/ do src/experiments/baseline.py ghi; script này chỉ tính lại để in bảng.
Hàm phân tích (cặp lỗi, xác suất, độ quan trọng theo khối) dùng chung ở src/evaluation/analysis.py,
giống hệt các mô hình khác; so với Fisher (H4) làm một lần cho mọi mô hình ở compare_models.

Quy tắc: mỗi thí nghiệm chỉ đổi MỘT yếu tố, giữ nguyên seed, chọn theo
VALIDATION. Tập test chỉ dùng MỘT lần ở E5, sau khi cấu hình đã chốt.

    E0  Mốc thấp nhất: tâm gần nhất                         (EDA 3.1)
    E1  Chuẩn hoá: không / vô hướng / từng pixel             (EDA 1; chốt chuẩn hoá vô hướng)
    E2a Learning rate × momentum (batch 128)                 (EDA 5.2: chọn learning rate)
    E2c Kích thước batch, mỗi batch dò η riêng              (chốt batch trên validation)
    E2b Tốc độ hội tụ: full-batch so với mini-batch đã chọn (EDA 2, 5.2: điều kiện số xấu)
    E3  Hệ số L2 λ và độ "lốm đốm" của ảnh trọng số          (EDA 1.1, 2.1, 5.2)
    E4  Chọn pixel: giữ hết / bỏ dead / bỏ hiếm / crop 24×24 (EDA 1.1, 1.3)
    E5  Mô hình cuối: train / val / test, xác suất, lỗi theo cặp, ảnh trọng số
                                                            (số liệu cho H1–H4 ở compare_models)
"""

import argparse
import json
import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import EDA_CLOSE_PAIRS, PERM_REPEATS, SEED
from src.data.features import make_features
from src.data.preprocess import load_splits
from src.evaluation.analysis import (block_permutation_importance, confident_error_records, pair_errors,
                                     probabilistic_scores, share_in_pairs, top_confused_pairs)
from src.evaluation.metrics import evaluate, summary_table
from src.evaluation.results_io import (Logger, compact_eval, format_confusion, model_dir, rel_path,
                                       save_final_metrics, write_json)
from src.models.logistic_regression import SoftmaxRegressionNP
from src.models.nearest_centroid import NearestCentroidNP
from src.models.pipeline import Pipeline
from src.plots.logistic_regression import plot_all

KEY = "logistic_regression"
OUT_DIR = model_dir(KEY)

# Siêu tham số mặc định trước khi chọn (E1 dùng, E2 trở đi thay bằng giá trị đã chọn)
DEFAULT = dict(mask="dead", scale="scalar", eta=0.003, momentum=0.9, batch_size=128, lam=1e-4)

FULL = dict(
    sweep_epochs=15,                                 # E1, E4
    grid_epochs=10,                                  # E2 lưới learning rate × momentum
    curve_epochs=100,                                # E2 đường hội tụ
    lam_epochs=60,                                   # E3: đủ dài để thấy tác dụng của L2
    final_epochs=150, patience=8,                    # E5: early stopping trên val
    etas=[0.001, 0.003, 0.01, 0.03, 0.1],
    momenta=[0.0, 0.5, 0.9],
    lams=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    batch_epochs=20,                                 # E2c
    # batch lớn cần η lớn hơn (cùng số epoch thì ít bước cập nhật hơn), nên mỗi batch có lưới η riêng
    batch_grid={32: [0.003, 0.01, 0.03], 64: [0.01, 0.03, 0.1], 128: [0.01, 0.03, 0.1],
                256: [0.03, 0.1, 0.3], 1024: [0.1, 0.3, 1.0], None: [0.3, 1.0, 3.0]},
)
QUICK = dict(FULL, sweep_epochs=2, grid_epochs=2, curve_epochs=5, lam_epochs=3, final_epochs=4, patience=2,
             batch_epochs=2, batch_grid={128: [0.03], 1024: [0.3], None: [1.0]},
             etas=[0.003, 0.03], momenta=[0.0, 0.9], lams=[0.0, 1e-3])

# Chênh val loss dưới mức này coi như bằng nhau (cỡ dao động giữa các seed của mini-batch);
# khi đó chọn cấu hình ít đặc trưng hơn (rẻ hơn, ít trọng số hơn).
VAL_LOSS_TOL = 0.002

# Ảnh trọng số mang NaN ở pixel bị bỏ; nanmean/nanstd trên cột toàn NaN chỉ báo cảnh báo, kết quả vẫn đúng (NaN)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="(Mean of empty slice|Degrees of freedom)")


# ---------------------------------------------------------------------------
# Hàm dùng chung
# ---------------------------------------------------------------------------
def make_model(eta, lam, epochs, momentum, batch_size, patience=None, eval_every=1):
    return SoftmaxRegressionNP(eta=eta, lam=lam, max_epochs=epochs, batch_size=batch_size,
                               momentum=momentum, patience=patience, seed=SEED, eval_every=eval_every, tol=None)


def fit_on(Z_tr, y_tr, Z_va, y_va, **kw):
    """Huấn luyện trên đặc trưng đã biến đổi; trả về (mô hình, dict số liệu tại epoch có val loss nhỏ nhất)."""
    m = make_model(**kw).fit(Z_tr, y_tr, Z_va, y_va)
    h = m.history_
    i = int(np.nanargmin(h["val_loss"])) if np.isfinite(h["val_loss"]).any() else len(h["val_loss"]) - 1
    row = {
        "best_epoch": h["epoch"][i], "val_acc": h["val_acc"][i], "val_loss": h["val_loss"][i],
        "train_acc": h["train_acc"][i], "train_loss": h["train_loss"][i],
        "last_val_acc": h["val_acc"][-1], "fit_time": m.fit_time_, "diverged": not np.isfinite(h["train_loss"][-1]),
    }
    return m, row


def fit_pipeline(splits, features, **kw):
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    t0 = time.perf_counter()
    Z_tr = features.fit_transform(X_tr)
    Z_va = features.transform(X_va)
    prep_time = time.perf_counter() - t0
    m, row = fit_on(Z_tr, y_tr, Z_va, y_va, **kw)
    row["prep_time"] = prep_time
    row["n_features"] = Z_tr.shape[1]
    return Pipeline(features, m), row


def pick_parsimonious(cands, tol=None):
    """cands: list (val loss, số đặc trưng, lựa chọn). Trả về lựa chọn ít đặc trưng nhất
    trong số các ứng viên có val loss ≤ val loss nhỏ nhất + tol."""
    tol = VAL_LOSS_TOL if tol is None else tol
    best = min(c[0] for c in cands)
    return min((c for c in cands if c[0] <= best + tol), key=lambda c: (c[1], c[0]))[2]


def weight_roughness(W):
    """Độ "lốm đốm" của ảnh trọng số: trung bình |w_p − w_{p kề}| chia trung bình |w_p|.

    Pixel kề nhau tương quan mạnh (EDA 2.1, r ≈ 0,71), nên trọng số không có L2 có thể nhận giá
    trị trái dấu ở hai pixel kề mà vẫn cho cùng dự đoán; L2 phạt kiểu nghiệm đó.
    """
    W = W - np.nanmean(W, axis=0, keepdims=True)    # bỏ phần chung giữa các lớp (softmax bất biến)
    dx = np.abs(np.diff(W, axis=2))
    dy = np.abs(np.diff(W, axis=1))
    return float((np.nanmean(dx) + np.nanmean(dy)) / 2 / np.nanmean(np.abs(W)))


# ---------------------------------------------------------------------------
# E0. Mốc thấp nhất: tâm gần nhất
# ---------------------------------------------------------------------------
def run_e0(splits, log):
    log("\n=== E0. Tâm gần nhất (mốc) ===")
    nc = NearestCentroidNP()
    res = evaluate(nc, {k: splits[k] for k in ("train", "val")})
    res["model"] = "Tâm gần nhất"
    log(f"val acc = {res['val']['accuracy']:.4f} | train acc = {res['train']['accuracy']:.4f}")
    return nc, res


# ---------------------------------------------------------------------------
# E1. Chuẩn hoá
# ---------------------------------------------------------------------------
def run_e1(splits, cfg, log):
    log("\n=== E1. Chuẩn hoá (mask = dead, mỗi cách chọn η tốt nhất trong 3 giá trị) ===")
    log("EDA chốt chuẩn hoá vô hướng (một μ, một σ); 'pixel' = z-score từng pixel, chỉ để đối chứng")
    X_va = splits["val"][0]
    rows = {}
    log(f"{'chuẩn hoá':>9} | {'η':>6} | {'max|z| val':>10} | {'val acc':>7} | {'val loss':>8} | {'sai mà p>0.99':>13}")
    for scale in ["none", "scalar", "pixel"]:
        best = None
        for eta in [0.001, 0.003, 0.01] if cfg is FULL else [0.003]:
            pipe, row = fit_pipeline(splits, make_features("dead", scale), eta=eta, lam=DEFAULT["lam"],
                                     epochs=cfg["sweep_epochs"], momentum=0.9, batch_size=128)
            if best is None or row["val_loss"] < best[1]["val_loss"]:
                best = (pipe, row, eta)
        pipe, row, eta = best
        z_max = float(np.abs(pipe.features.transform(X_va)).max())
        proba = pipe.predict_proba(X_va)
        ps = probabilistic_scores(splits["val"][1], proba)
        row.update(eta=eta, max_abs_z=z_max, **ps)
        rows[scale] = row
        log(f"{scale:>9} | {eta:6.3f} | {z_max:10.1f} | {row['val_acc']:7.4f} | {row['val_loss']:8.4f} | "
            f"{ps['overconfident_errors']:13.4%}")
    return rows


# ---------------------------------------------------------------------------
# E2. Learning rate, momentum, tốc độ hội tụ
# ---------------------------------------------------------------------------
def run_e2(splits, cfg, log, scale):
    log(f"\n=== E2a. Learning rate × momentum (mini-batch 128, {cfg['grid_epochs']} epoch, scale = {scale}) ===")
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    feats = make_features("dead", scale)
    Z_tr, Z_va = feats.fit_transform(X_tr), feats.transform(X_va)

    etas, moms = cfg["etas"], cfg["momenta"]
    acc = np.full((len(moms), len(etas)), np.nan)
    loss = np.full_like(acc, np.nan)
    for i, a in enumerate(moms):
        for j, eta in enumerate(etas):
            _, row = fit_on(Z_tr, y_tr, Z_va, y_va, eta=eta, lam=DEFAULT["lam"], epochs=cfg["grid_epochs"],
                            momentum=a, batch_size=128)
            if not row["diverged"]:
                acc[i, j], loss[i, j] = row["val_acc"], row["val_loss"]
    log("val acc (hàng: momentum α, cột: η); η_eff = η / (1 − α)")
    log(f"{'':>8}" + "".join(f"{e:>9}" for e in etas))
    for i, a in enumerate(moms):
        log(f"α = {a:<4}" + "".join(f"{v:9.4f}" for v in acc[i]))
    i, j = np.unravel_index(np.nanargmin(loss), loss.shape)
    best = dict(momentum=moms[i], eta=etas[j])
    log(f"Chọn theo val loss nhỏ nhất: α = {best['momentum']}, η = {best['eta']} "
        f"(val acc {acc[i, j]:.4f}, val loss {loss[i, j]:.4f})")

    return best, {"grid_acc": acc, "grid_loss": loss, "etas": etas, "momenta": moms, "epochs": cfg["grid_epochs"]}


# ---------------------------------------------------------------------------
# E2c. Kích thước batch
# ---------------------------------------------------------------------------
def run_e2c(splits, cfg, log, momentum):
    """Mỗi batch size dò η riêng (cùng α); chọn cặp (batch, η) có val loss nhỏ nhất."""
    log(f"\n=== E2c. Kích thước batch ({cfg['batch_epochs']} epoch, α = {momentum}, mỗi batch dò η riêng) ===")
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    feats = make_features("dead", "scalar")
    Z_tr, Z_va = feats.fit_transform(X_tr), feats.transform(X_va)
    N = len(Z_tr)
    rows = []
    log(f"{'batch':>6} | {'η tốt nhất':>10} | {'bước/epoch':>10} | {'val acc':>7} | {'val loss':>8} | {'s/epoch':>7}")
    for bs, etas in cfg["batch_grid"].items():
        best = None
        for eta in etas:
            _, row = fit_on(Z_tr, y_tr, Z_va, y_va, eta=eta, lam=DEFAULT["lam"], epochs=cfg["batch_epochs"],
                            momentum=momentum, batch_size=bs)
            if not row["diverged"] and (best is None or row["val_loss"] < best["val_loss"]):
                best = dict(row, eta=eta)
        if best is None:                                     # mọi η đều phân kỳ
            log(f"{str(bs or 'full'):>6} | phân kỳ với mọi η thử")
            continue
        best.update(batch_size=bs, steps_per_epoch=-(-N // (bs or N)),
                    sec_per_epoch=best["fit_time"] / cfg["batch_epochs"])
        rows.append(best)
        log(f"{str(bs or 'full'):>6} | {best['eta']:10.3g} | {best['steps_per_epoch']:10d} | {best['val_acc']:7.4f} | "
            f"{best['val_loss']:8.4f} | {best['sec_per_epoch']:7.3f}")
    # Quy tắc: giữ batch mặc định nếu nó cách val loss tốt nhất dưới VAL_LOSS_TOL (chênh lệch cỡ nhiễu),
    # chỉ đổi khi batch khác tốt hơn rõ rệt. Cùng tinh thần với quy tắc chọn mask ở E4.
    best = min(rows, key=lambda r: r["val_loss"])
    near = [r for r in rows if r["val_loss"] <= best["val_loss"] + VAL_LOSS_TOL]
    default = [r for r in near if r["batch_size"] == DEFAULT["batch_size"]]
    chosen = default[0] if default else best
    reason = (f"mặc định, cách tốt nhất (batch {best['batch_size'] or 'full'}) < {VAL_LOSS_TOL}" if default
              else "val loss nhỏ nhất, tốt hơn mặc định rõ rệt")
    log(f"Chọn batch = {chosen['batch_size'] or 'full'}, η = {chosen['eta']} ({reason}). "
        f"Các batch cách tốt nhất < {VAL_LOSS_TOL}: {', '.join(str(r['batch_size'] or 'full') for r in near)}")
    choice = dict(batch_size=chosen["batch_size"], eta=chosen["eta"])
    return choice, {"rows": rows, "chosen": choice, "momentum": momentum, "epochs": cfg["batch_epochs"],
                    "n_train": N, "tol": VAL_LOSS_TOL}


# ---------------------------------------------------------------------------
# E2b. Tốc độ hội tụ
# ---------------------------------------------------------------------------
def run_e2b(splits, cfg, log, opt):
    """Đường J(W) theo epoch: gradient descent full-batch so với cấu hình mini-batch đã chọn."""
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    log(f"\n=== E2b. Tốc độ hội tụ của J(W) ({cfg['curve_epochs']} epoch) ===")
    n_ep = cfg["curve_epochs"]
    runs = {
        "full-batch, η=0.2, α=0, vô hướng": ("scalar", dict(eta=0.2, momentum=0.0, batch_size=None)),
        "full-batch, η=0.2, α=0.9, vô hướng": ("scalar", dict(eta=0.2, momentum=0.9, batch_size=None)),
        f"mini-batch {opt['batch_size']}, η={opt['eta']}, α={opt['momentum']}, vô hướng": (
            "scalar", dict(eta=opt["eta"], momentum=opt["momentum"], batch_size=opt["batch_size"])),
    }
    curves = {}
    for name, (sc, kw) in runs.items():
        f = make_features("dead", sc)
        m = make_model(lam=DEFAULT["lam"], epochs=n_ep, **kw).fit(f.fit_transform(X_tr), y_tr, f.transform(X_va), y_va)
        curves[name] = m.history_
        log(f"{name:>42}: J sau 10 epoch = {m.history_['train_loss'][min(9, n_ep - 1)]:.4f} | "
            f"J cuối = {m.history_['train_loss'][-1]:.4f} | val acc cuối = {m.history_['val_acc'][-1]:.4f} | "
            f"{m.fit_time_:.1f}s")
    return curves


# ---------------------------------------------------------------------------
# E3. L2
# ---------------------------------------------------------------------------
def run_e3(splits, cfg, log, opt):
    log(f"\n=== E3. Hệ số L2 λ ({cfg['lam_epochs']} epoch, η = {opt['eta']}, α = {opt['momentum']}, "
        f"batch = {opt['batch_size']}) ===")
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    feats = make_features("dead", "scalar")
    Z_tr, Z_va = feats.fit_transform(X_tr), feats.transform(X_va)
    rows, images = {}, {}
    log(f"{'λ':>7} | {'train loss':>10} | {'val loss':>8} | {'chênh':>6} | {'val acc':>7} | {'lốm đốm':>7} | {'‖W‖':>7}")
    for lam in cfg["lams"]:
        m, row = fit_on(Z_tr, y_tr, Z_va, y_va, lam=lam, epochs=cfg["lam_epochs"], **opt)
        W = Pipeline(feats, m).pixel_weights()
        # train loss ở đây là cross-entropy thuần, để so trực tiếp với val loss
        ce_tr = row["train_loss"] - 0.5 * lam * float(np.sum(m.weights_.astype(np.float64) ** 2))
        row.update(lam=lam, train_ce=ce_tr, gap=row["val_loss"] - ce_tr, roughness=weight_roughness(W),
                   w_norm=float(np.linalg.norm(m.weights_)))
        rows[lam] = row
        images[str(lam)] = W.reshape(10, -1)
        log(f"{lam:7.0e} | {ce_tr:10.4f} | {row['val_loss']:8.4f} | {row['gap']:6.4f} | {row['val_acc']:7.4f} | "
            f"{row['roughness']:7.3f} | {row['w_norm']:7.2f}")
    best_lam = min(rows, key=lambda k: rows[k]["val_loss"])
    log(f"Chọn λ = {best_lam:g} (val loss nhỏ nhất)")
    return best_lam, rows, images


# ---------------------------------------------------------------------------
# E4. Chọn pixel
# ---------------------------------------------------------------------------
def run_e4(splits, cfg, log, hp):
    log(f"\n=== E4. Chọn pixel ({cfg['sweep_epochs']} epoch) ===")
    rows = {}
    log(f"{'mask':>7} | {'số đặc trưng':>12} | {'fit (s)':>7} | {'val acc':>7} | {'val loss':>8}")
    for mask in ["none", "dead", "rare", "crop24"]:
        _, row = fit_pipeline(splits, make_features(mask, "scalar"), epochs=cfg["sweep_epochs"], **hp)
        rows[mask] = row
        log(f"{mask:>7} | {row['n_features']:12d} | {row['fit_time']:7.2f} | {row['val_acc']:7.4f} | {row['val_loss']:8.4f}")
    return rows


# ---------------------------------------------------------------------------
# E5. Mô hình cuối
# ---------------------------------------------------------------------------
def run_e5(splits, cfg, log, final_cfg, nc, nc_res, n_jobs):
    log("\n=== E5. Mô hình cuối (early stopping trên val; test dùng một lần) ===")
    log("Cấu hình:", json.dumps(final_cfg, ensure_ascii=False))
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    feats = make_features(final_cfg["mask"], "scalar")
    model = make_model(eta=final_cfg["eta"], lam=final_cfg["lam"], epochs=final_cfg["max_epochs"],
                       momentum=final_cfg["momentum"], batch_size=final_cfg["batch_size"],
                       patience=final_cfg["patience"])
    pipe = Pipeline(feats, model, name="Logistic Regression")
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr, X_va, y_va)
    wall = time.perf_counter() - t0
    # Định nghĩa chung (results/README.md): gồm tiền xử lý, cập nhật và val loss cho early stopping;
    # không gồm loss/accuracy trên train mỗi epoch (chỉ để vẽ đường học).
    fit_time = wall - model.diag_time_
    log(f"Dừng ở epoch {model.n_epochs_}, chọn epoch {model.best_epoch_} | fit {fit_time:.1f}s "
        f"(cập nhật {model.fit_time_:.1f}s, val loss {model.val_time_:.1f}s; kể cả chẩn đoán: {wall:.1f}s)")

    res = evaluate(pipe, splits, fit=False)                  # train, val, test - test chỉ ở đây
    res["model"] = "Logistic Regression"
    res["fit_time"] = fit_time
    nc_full = evaluate(nc, splits, fit=False)
    nc_full["model"] = "Tâm gần nhất"
    nc_full["fit_time"] = nc_res.get("fit_time", float("nan"))           # bản ghi chính thức: baseline.py
    for split in ("train", "val", "test"):
        log(f"\n[{split}]")
        log(summary_table([nc_full, res], split))

    # --- xác suất ---
    log("\nĐộ đo xác suất (để so với Multinomial NB):")
    prob = {}
    for split in ("train", "val", "test"):
        X, y = splits[split]
        prob[split] = probabilistic_scores(y, pipe.predict_proba(X))
        p = prob[split]
        log(f"  {split:>5}: log-loss {p['log_loss']:.4f} | sai mà p>0.99: "
            f"{p['overconfident_errors']:.3%} mẫu ({p['overconfident_share_of_errors']:.1%} số lỗi) | "
            f"độ tự tin TB {p['mean_confidence']:.3f}")

    # --- lỗi theo cặp ---
    log("\nLỗi theo cặp lớp (cộng hai chiều). Dấu * đánh dấu 5 cặp EDA 3.1 dự báo:")
    pairs_out = {}
    for split in ("val", "test"):
        for name, r in [("LR", res), ("Tâm gần nhất", nc_full)]:
            cm = r[split]["confusion_matrix"]
            top = top_confused_pairs(cm, 8)
            share = share_in_pairs(cm)
            pairs_out[f"{name}/{split}"] = {"top": [[[i, j], n] for i, j, n in top], "share_eda_pairs": share,
                                            "eda_pairs": {f"{i}-{j}": n for (i, j), n in pair_errors(cm, EDA_CLOSE_PAIRS).items()}}
            txt = ", ".join(f"{i}–{j}{'*' if (i, j) in EDA_CLOSE_PAIRS else ''}: {n}" for i, j, n in top)
            log(f"  {name:>12} {split:>4}: {txt}")
            log(f"  {'':>12} {'':>4}  5 cặp EDA chiếm {share:.1%} tổng lỗi (nếu rải đều: {5 / 45:.1%})")

    # --- ma trận nhầm lẫn trên cả 3 tập (yêu cầu bắt buộc của đề) ---
    log("\nMa trận nhầm lẫn của Logistic Regression (hàng: nhãn thật, cột: dự đoán; cột cuối: recall của lớp):")
    for split in ("train", "val", "test"):
        log(format_confusion(res[split]["confusion_matrix"], f"\n[{split}]"))

    # --- ảnh trọng số; so với Fisher (H4) làm một lần cho mọi mô hình ở compare_models ---
    W = pipe.pixel_weights().reshape(10, -1)                 # (10, 784), NaN ở pixel bị bỏ
    t0 = time.perf_counter()
    blocks = block_permutation_importance(pipe, X_va, y_va, n_repeats=PERM_REPEATS, seed=SEED)
    log(f"\nSố đo chung cho mọi mô hình: hoán vị khối 4×4 trên val, khối quan trọng nhất làm giảm acc "
        f"{blocks.max():.4f} ({time.perf_counter() - t0:.0f}s)")

    X_te, y_te = splits["test"]
    errors = confident_error_records(y_te, pipe.predict_proba(X_te))

    # --- kết quả theo cấu trúc chung (results/README.md), đầu vào của compare_models ---
    save_final_metrics(
        KEY, "Logistic Regression", res, final_cfg, probabilistic=prob,
        block_importance=blocks, n_jobs=n_jobs,
        extra={"best_epoch": model.best_epoch_, "stopped_epoch": model.n_epochs_, "update_time": model.fit_time_,
               "val_loss_time": model.val_time_, "fit_time_with_diagnostics": wall}, quick=cfg is QUICK)

    return {
        "config": final_cfg, "best_epoch": model.best_epoch_, "fit_time": fit_time,
        "logistic_regression": compact_eval(res), "nearest_centroid": compact_eval(nc_full),
        "probabilistic": prob, "pairs": pairs_out,
        "history": model.history_, "weights": W, "nearest_centroid_weights": nc.weights_.T,
        "confident_errors": errors,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="ít epoch, chỉ để kiểm tra code")
    ap.add_argument("--blas-threads", type=int, default=8, help="số luồng BLAS (mặc định 8, xem src/experiments/mlp.py)")
    args = ap.parse_args()
    with threadpool_limits(args.blas_threads, user_api="blas"):
        experiment(args.quick, args.blas_threads)


def experiment(quick, n_threads):
    cfg = QUICK if quick else FULL
    t_start = time.perf_counter()
    log = Logger(OUT_DIR / "log.txt")
    splits = load_splits()
    log(f"train {splits['train'][0].shape} | val {splits['val'][0].shape} | test {splits['test'][0].shape}"
        f" | {n_threads} luồng BLAS")

    nc, nc_res = run_e0(splits, log)
    e1 = run_e1(splits, cfg, log)
    opt, e2 = run_e2(splits, cfg, log, "scalar")            # E2a: η, α ở batch 128
    batch_choice, e2c = run_e2c(splits, cfg, log, opt["momentum"])
    opt = dict(opt, **batch_choice)                           # batch và η chốt ở E2c
    e2b = run_e2b(splits, cfg, log, opt)
    best_lam, e3, e3_weights = run_e3(splits, cfg, log, opt)
    hp = dict(eta=opt["eta"], momentum=opt["momentum"], batch_size=opt["batch_size"], lam=best_lam)
    e4 = run_e4(splits, cfg, log, hp)
    best_mask = pick_parsimonious([(r["val_loss"], r["n_features"], m) for m, r in e4.items()])
    log(f"Chọn mask = {best_mask} (ít đặc trưng nhất trong các cách có val loss cách tốt nhất < {VAL_LOSS_TOL})")
    # Ghi đủ mọi siêu tham số của mô hình cuối, kể cả early stopping
    final_cfg = dict(scale="scalar", lam=best_lam, momentum=hp["momentum"], batch_size=hp["batch_size"],
                     max_epochs=cfg["final_epochs"], patience=cfg["patience"], mask=best_mask, eta=hp["eta"])
    e5 = run_e5(splits, cfg, log, final_cfg, nc, nc_res, n_threads)

    log(f"\nTổng thời gian: {time.perf_counter() - t_start:.0f}s")
    summary = {"quick": quick, "e1_scaling": e1, "e2": e2, "e2c_batch": e2c, "e2_choice": opt,
               "e2b_convergence": e2b, "e3_l2": {str(k): v for k, v in e3.items()}, "e3_weights": e3_weights,
               "e4_mask": e4, "final": e5}
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'summary.json')}, log.txt và figures/")


if __name__ == "__main__":
    main()