"""Thí nghiệm CNN tự cài (src/models/cnn.py) trên MNIST: bước 3 "Đổi sang CNN" của slide khuyến nghị bài 06.

Chạy từ gốc repo:
    python -m src.experiments.cnn --quick        # kiểm tra code chạy được (vài phút, train 10.000 ảnh)
    python -m src.experiments.cnn                # bản đầy đủ (khoảng 20 phút trên CPU)

Quy tắc chung như các mô hình khác: mỗi thí nghiệm đổi một yếu tố, chọn theo validation, test dùng
đúng một lần ở E4. Chuẩn hoá vô hướng học từ splits["train"]; không bỏ pixel vì CNN cần đủ lưới 28×28.
Lý do làm CNN (EDA 5.4): thông tin cục bộ (tương quan ≈ 0 từ khoảng cách 5), cặp gần nhau khác ở vùng nhỏ,
vị trí nét lệch vài pixel trong cùng lớp; so với MLP về accuracy và lỗi ở 5 cặp gần nhau.

    E0  Mốc: đọc final_metrics.json của các mô hình đã chạy (mốc phải vượt: RF ≈ 0,97)
    E1  Learning rate của Adam, cùng 3 epoch như slide
    E2  Độ rộng (số bộ lọc mỗi khối conv): accuracy so với thời gian
    E3  Số epoch: huấn luyện cấu hình cuối với dừng sớm theo val loss, vẽ đường học
    E4  Đánh giá cuối: train/val/test, lỗi theo cặp, độ quan trọng

Kết quả ghi vào results/cnn/ theo quy ước chung (results/README.md):
    log.txt, summary.json, final_metrics.json
    figures/   hp_e1_lr.png, hp_e2_width.png, eval_learning_curve.png, eval_confusion.png,
               eval_confident_errors.png: do src/plots/cnn.py vẽ từ summary.json và final_metrics.json
               (gọi ở cuối script; vẽ lại riêng: python -m src.plots.cnn)
"""

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import DISPLAY_NAMES, EDA_CLOSE_PAIRS, MODEL_KEYS, PERM_REPEATS, SEED
from src.data.features import make_features
from src.data.preprocess import load_splits, subset_train
from src.evaluation.analysis import (block_permutation_importance, confident_error_records, log_loss, pair_errors,
                                     probabilistic_scores, share_in_pairs, top_confused_pairs)
from src.evaluation.metrics import classification_scores, evaluate
from src.evaluation.results_io import (Logger, format_confusion, load_final_metrics, model_dir, rel_path,
                                       save_final_metrics, write_json)
from src.models.cnn import CNNClassifierNP
from src.models.nearest_centroid import NearestCentroidNP
from src.models.pipeline import Pipeline
from src.plots.cnn import plot_all

KEY = "cnn"
OUT_DIR = model_dir(KEY)

# Cấu hình xuất phát: đúng ví dụ slide 12 (Adam 1e-3, 3 epoch, 16 → 32 bộ lọc, 128 nơron ẩn)
BASE_CONFIG = dict(scale="scalar", channels=(16, 32), hidden=128, lr=1e-3, batch_size=128, max_epochs=3,
                   patience=None)
# Hai cấu hình cách nhau dưới ngưỡng này trên val (10.000 ảnh: 0,1 điểm = 10 ảnh) coi như bằng nhau
TOL = 0.001


# ---------------------------------------------------------------- tiện ích
def make_cnn(cfg, seed=SEED, verbose=False):
    model = CNNClassifierNP(channels=cfg["channels"], hidden=cfg["hidden"], lr=cfg["lr"],
                            batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"], patience=cfg["patience"],
                            seed=seed, verbose=verbose)
    return Pipeline(make_features("none", cfg["scale"]), model, name="CNN")


def run_cnn(cfg, splits, seed=SEED, keep=False):
    """Huấn luyện trên train (val cho dừng sớm / đường học), trả về độ đo trên val (không đụng test)."""
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    pipe = make_cnn(cfg, seed)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr, X_va, y_va)
    fit_s = time.perf_counter() - t0
    m = pipe.model
    best = m.history_[m.best_epoch_ - 1]
    row = dict(cfg, seed=seed, val_acc=best["val_acc"], val_loss=best["val_loss"], fit_s=fit_s,
               epochs=len(m.history_), best_epoch=m.best_epoch_, n_params=m.n_params_, history=m.history_)
    if keep:
        row["_pipe"] = pipe
    return row


# ---------------------------------------------------------------- các thí nghiệm
def run_e0(log):
    """Mốc so sánh: các mô hình đã có final_metrics.json."""
    log("\n=== E0. Mốc so sánh (đọc results/<mô hình>/final_metrics.json) ===")
    out = {}
    for key in ("nearest_centroid", *[k for k in MODEL_KEYS if k != KEY]):
        fm = load_final_metrics(key)
        if fm is None:
            log(f"{DISPLAY_NAMES.get(key, key):<26}: chưa có kết quả")
            continue
        out[key] = {s: fm["splits"][s]["accuracy"] for s in ("val", "test")}
        log(f"{DISPLAY_NAMES.get(key, key):<26}: val acc = {out[key]['val']:.4f} | test acc = {out[key]['test']:.4f}"
            f" | fit {fm['fit_time']:.1f} s")
    return out


def run_e1(splits, cfg, log, lr_grid):
    log(f"\n=== E1. Learning rate của Adam ({cfg['max_epochs']} epoch, không dừng sớm) ===")
    rows = []
    for lr in lr_grid:
        r = run_cnn(dict(cfg, lr=lr), splits)
        last = r["history"][-1]
        rows.append(dict(r, val_acc_last=last["val_acc"]))
        log(f"lr = {lr:g}: val acc (epoch cuối) = {last['val_acc']:.4f} | val loss = {last['val_loss']:.4f} | "
            f"train loss = {last['train_loss']:.4f} | {r['fit_s']:.0f} s")
    chosen = max(rows, key=lambda r: r["val_acc_last"])["lr"]
    log(f"Chọn lr = {chosen:g}")
    return dict(rows=rows, chosen=chosen)


def run_e2(splits, cfg, log, width_grid):
    log(f"\n=== E2. Độ rộng: số bộ lọc mỗi khối conv ({cfg['max_epochs']} epoch, lr = {cfg['lr']:g}) ===")
    rows = []
    for ch in width_grid:
        r = run_cnn(dict(cfg, channels=tuple(ch)), splits)
        r["val_acc_last"] = r["history"][-1]["val_acc"]
        rows.append(r)
        n_params = f"{r['n_params']:>7,}".replace(",", ".")
        log(f"channels = {str(tuple(ch)):<9}: val acc = {r['val_acc_last']:.4f} | {n_params} tham số | "
            f"{r['fit_s']:.0f} s")
    best = max(r["val_acc_last"] for r in rows)
    ok = [r for r in rows if r["val_acc_last"] >= best - TOL]
    chosen = min(ok, key=lambda r: r["fit_s"])["channels"]
    log(f"Chọn channels = {chosen} (nhanh nhất trong các cấu hình cách tốt nhất không quá {TOL})")
    return dict(rows=rows, chosen=list(chosen), tol=TOL)


def run_e3(splits, cfg, log):
    log(f"\n=== E3. Số epoch: tối đa {cfg['max_epochs']}, dừng sớm khi val loss không giảm {cfg['patience']} epoch ===")
    log("Cấu hình: " + json.dumps(cfg, ensure_ascii=False))
    r = run_cnn(cfg, splits, keep=True)
    for h in r["history"]:
        log(f"epoch {h['epoch']:>2}: train loss {h['train_loss']:.4f} acc {h['train_acc']:.4f} | "
            f"val loss {h['val_loss']:.4f} acc {h['val_acc']:.4f} | {h['seconds']:.0f} s")
    log(f"Dừng sau {r['epochs']} epoch, khôi phục trọng số epoch {r['best_epoch']} (val loss nhỏ nhất) | "
        f"huấn luyện {r['fit_s']:.0f} s")
    return r


def run_e4(splits, cfg, r3, log, perm_repeats, n_jobs, quick=False, perm_n=None):
    """Đánh giá cuối trên train/val/test (test dùng một lần), cùng các phân tích chung với RF, LR, NB."""
    log("\n=== E4. Đánh giá cuối (test dùng một lần) ===")
    pipe = r3["_pipe"]
    res = evaluate(pipe, splits, fit=False)
    # fit_time theo định nghĩa chung: gồm val loss mỗi epoch vì dừng sớm cần nó
    res["fit_time"] = r3["fit_s"]
    log(f"Huấn luyện: {res['fit_time']:.1f} s | {pipe.model.n_params_:,} tham số | epoch giữ lại {r3['best_epoch']}"
        .replace(",", "."))
    for name in ("train", "val", "test"):
        s = res[name]
        log(f"{name:>5}: acc {s['accuracy']:.4f} | P {s['precision_macro']:.4f} | R {s['recall_macro']:.4f} | "
            f"F1 {s['f1_macro']:.4f} | dự đoán {s['predict_time']:.2f} s")
    for name in ("train", "val", "test"):
        log(format_confusion(res[name]["confusion_matrix"], f"\n[{name}] ma trận nhầm lẫn (hàng: nhãn thật, cột: dự đoán)"))

    out = dict(config=cfg, fit_s=res["fit_time"], best_epoch=r3["best_epoch"], history=r3["history"])
    nc = NearestCentroidNP().fit(*splits["train"])
    prob, P_all = {}, {}
    for name in ("train", "val", "test"):
        X, y = splits[name]
        P = P_all[name] = pipe.predict_proba(X)
        prob[name] = probabilistic_scores(y, P)
        cm = res[name]["confusion_matrix"]
        cm_nc = classification_scores(y, nc.predict(X))["confusion_matrix"]
        out[name] = dict(accuracy=res[name]["accuracy"], f1_macro=res[name]["f1_macro"], log_loss=log_loss(y, P),
                         pairs=pair_errors(cm, EDA_CLOSE_PAIRS), share_close_pairs=share_in_pairs(cm),
                         top_pairs=top_confused_pairs(cm, 5), n_errors=int(cm.sum() - np.trace(cm)),
                         nearest_centroid=dict(accuracy=float(np.trace(cm_nc) / cm_nc.sum()),
                                               pairs=pair_errors(cm_nc, EDA_CLOSE_PAIRS)))
    for name in ("val", "test"):
        o, p = out[name], prob[name]
        log(f"[{name}] log-loss {o['log_loss']:.4f} | sai mà p > 0,99: {p['overconfident_errors']:.2%} | "
            f"p trung bình của mẫu sai: {p['mean_confidence_errors']:.3f}")
        log(f"[{name}] Lỗi ở 5 cặp gần nhất chiếm {o['share_close_pairs']:.1%} số lỗi (rải đều: {5 / 45:.1%}); "
            f"lỗi theo cặp CNN {o['pairs']}")
        log(f"[{name}]     5 cặp nhầm nhiều nhất: {o['top_pairs']}")
        n_nc = (1 - o["nearest_centroid"]["accuracy"]) * len(splits[name][1])
        red = {k: 1 - o["pairs"][k] / max(o["nearest_centroid"]["pairs"][k], 1) for k in o["pairs"]}
        log(f"[{name}] So với tâm gần nhất: acc {o['nearest_centroid']['accuracy']:.4f}; CNN giảm lỗi "
            f"{1 - o['n_errors'] / n_nc:.1%} tổng, theo cặp " + ", ".join(f"{k}: {v:.1%}" for k, v in red.items()))

    # Độ quan trọng: hoán vị khối 4×4 (chung cho mọi mô hình); so với Fisher (H4) làm một lần ở compare_models.
    X_va, y_va = splits["val"]
    t0 = time.perf_counter()
    n = perm_n or len(X_va)
    drops = block_permutation_importance(pipe, X_va[:n], y_va[:n], n_repeats=perm_repeats, seed=SEED)
    log(f"Hoán vị khối 4×4 trên val ({time.perf_counter() - t0:.0f} s, {n} ảnh): khối quan trọng nhất làm giảm "
        f"acc {drops.max():.4f}")

    save_final_metrics(
        KEY, "CNN", res, cfg, probabilistic=prob,
        block_importance=drops, n_jobs=n_jobs,
        extra=dict(best_epoch=r3["best_epoch"], epochs_run=len(r3["history"]), n_params=pipe.model.n_params_,
                   history=r3["history"]),
        quick=quick)
    out["confident_errors"] = confident_error_records(splits["test"][1], P_all["test"])
    return out


# ---------------------------------------------------------------- chương trình chính
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="train 10.000 ảnh, ít epoch: chỉ để kiểm tra code")
    ap.add_argument("--blas-threads", type=int, default=8, help="số luồng BLAS (mặc định 8, xem src/experiments/mlp.py)")
    args = ap.parse_args()
    with threadpool_limits(args.blas_threads, user_api="blas"):
        experiment(args.quick, args.blas_threads)


def experiment(q, n_threads):
    log = Logger(OUT_DIR / "log.txt")
    t0 = time.perf_counter()
    splits = load_splits()
    search = subset_train(splits, 10_000, SEED) if q else splits
    log(f"train {splits['train'][0].shape} | val {splits['val'][0].shape} | test {splits['test'][0].shape}"
        f" | {n_threads} luồng BLAS | quick = {q}")

    cfg = dict(BASE_CONFIG, max_epochs=1 if q else BASE_CONFIG["max_epochs"])
    summary = {"base_config": dict(cfg), "E0": run_e0(log)}

    e1 = run_e1(search, cfg, log, (1e-3, 3e-3) if q else (3e-4, 1e-3, 3e-3))
    cfg["lr"] = e1["chosen"]
    summary["E1"] = e1

    e2 = run_e2(search, cfg, log, ((8, 16), (16, 32)) if q else ((8, 16), (16, 32), (32, 64)))
    cfg["channels"] = tuple(e2["chosen"])
    summary["E2"] = e2

    cfg.update(max_epochs=2 if q else 20, patience=1 if q else 3)
    r3 = run_e3(search if q else splits, cfg, log)
    summary["E3"] = {k: v for k, v in r3.items() if k != "_pipe"}
    cfg["channels"] = list(cfg["channels"])
    summary["final"] = run_e4(splits, cfg, r3, log, perm_repeats=1 if q else PERM_REPEATS, n_jobs=n_threads, quick=q,
                              perm_n=2000 if q else None)
    summary["total_s"] = time.perf_counter() - t0
    log(f"\nTổng thời gian: {summary['total_s'] / 60:.1f} phút")
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'summary.json')}, log.txt và figures/")


if __name__ == "__main__":
    main()
