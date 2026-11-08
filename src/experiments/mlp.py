"""Thí nghiệm MLP trên MNIST: dò siêu tham số E1–E6, so ứng viên E7, đánh giá cuối E8.

Chạy từ gốc repo:
    python -m src.experiments.mlp --quick        # kiểm tra code chạy được (train 10.000 ảnh, vài phút)
    python -m src.experiments.mlp                # bản đầy đủ

Quy tắc chung: mỗi thí nghiệm đổi một yếu tố so với baseline, chọn theo validation, test dùng một lần ở E8.
Chuẩn hoá vô hướng (EDA 5.4) học từ splits["train"], qua Pipeline.

    E0  Baseline 784-256-128-10, ReLU, He, γ = 0,05, α = 0,9, batch 128, 10 epoch (Lab 04: val ≥ 0,97)
    E1  learning rate γ × momentum α, γ_eff = γ / (1 − α)            (Bài 4 tr. 66–67)
    E2  kích thước batch
    E3  độ rộng và độ sâu
    E4  hàm kích hoạt và vanishing gradient (giải thích, không chọn) (Bài 4 tr. 39–41)
    E5  khởi tạo: đối xứng, Xavier hay He cho ReLU (giải thích)     (Bài 4 tr. 73–74)
    E6  điều chuẩn L2 và dropout, 15 epoch                          (EDA 5.4)
    E7  ba ứng viên A/B/C với early stopping, chọn theo val accuracy
    E8  đánh giá cuối: train/val/test, lỗi theo cặp, độ quan trọng; giải thích trọng số lớp ẩn 1

Kết quả ghi vào results/mlp/ theo quy ước chung (results/README.md):
    log.txt, summary.json, final_metrics.json
    figures/   hp_e1_lr_momentum.png … hp_e6_regularization.png, eval_learning_curve.png, eval_confusion.png,
               eval_confident_errors.png, interp_w1.png: do src/plots/mlp.py vẽ từ summary.json và
               final_metrics.json (gọi ở cuối script; vẽ lại riêng: python -m src.plots.mlp)

OpenBLAS mặc định mở nhiều luồng hơn số lõi; với ma trận cỡ mini-batch, việc đó làm một phép nhân chậm
hơn hàng chục lần (đo trên máy 14 lõi: 86 ms ở mặc định, 1,4 ms khi giới hạn 4 luồng). Script giới hạn
số luồng BLAS bằng --blas-threads (mặc định 8) và ghi con số này vào runtime.n_jobs.
"""

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import EDA_CLOSE_PAIRS, PERM_REPEATS, SEED
from src.data.features import make_features
from src.data.preprocess import load_splits, subset_train
from src.evaluation.analysis import (block_permutation_importance, confident_error_records, log_loss, pair_errors,
                                     probabilistic_scores, share_in_pairs, top_confused_pairs)
from src.evaluation.metrics import classification_scores, evaluate
from src.evaluation.results_io import Logger, format_confusion, model_dir, rel_path, save_final_metrics, write_json
from src.models.mlp import (MLPClassifierNP, backward, build_mlp, forward_with_cache, linears,
                            softmax_cross_entropy)
from src.models.nearest_centroid import NearestCentroidNP
from src.models.pipeline import Pipeline
from src.plots.mlp import plot_all

KEY = "mlp"
OUT_DIR = model_dir(KEY)

# Baseline giống Lab 04
BASE = dict(hidden=(256, 128), activation="relu", init="he", lr=0.05, momentum=0.9, batch_size=128,
            weight_decay=0.0, p_drop=0.0)
LRS = (0.005, 0.01, 0.05, 0.1, 0.5)
MOMENTA = (0.0, 0.5, 0.9)
BATCH_SIZES = (16, 64, 128, 256, 1024)
ARCHS = ((64,), (256,), (1024,), (256, 128), (256, 256, 256, 256))
REGS = {"không điều chuẩn": {}, "L2 λ=1e-4": {"weight_decay": 1e-4}, "L2 λ=1e-3": {"weight_decay": 1e-3},
        "dropout p=0.2": {"p_drop": 0.2}, "dropout p=0.5": {"p_drop": 0.5}}
FINAL_EPOCHS, FINAL_PATIENCE = 50, 5


# ---------------------------------------------------------------- tiện ích
def make_mlp(cfg, max_epochs, patience=None):
    model = MLPClassifierNP(hidden=tuple(cfg["hidden"]), activation=cfg["activation"], init=cfg["init"],
                            lr=cfg["lr"], momentum=cfg["momentum"], batch_size=cfg["batch_size"],
                            weight_decay=cfg["weight_decay"], p_drop=cfg["p_drop"], max_epochs=max_epochs,
                            patience=patience, seed=SEED)
    return Pipeline(make_features("none", "scalar"), model, name="MLP")


def run(splits, epochs, patience=None, **changes):
    """Huấn luyện baseline với các thay đổi trong changes (val cho đường học / early stopping)."""
    cfg = dict(BASE, **changes)
    pipe = make_mlp(cfg, epochs, patience)
    t0 = time.perf_counter()
    pipe.fit(*splits["train"], *splits["val"])
    pipe.wall_time_ = time.perf_counter() - t0
    return pipe, pipe.model.history_


def last_val_acc(h):
    return h["val_acc"][-1] if np.isfinite(h["train_loss"][-1]) else float("nan")


def cfg_json(cfg):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}


# ---------------------------------------------------------------- các thí nghiệm
def run_e0(splits, log, epochs):
    log(f"\n=== E0. Baseline 784-256-128-10 ({epochs} epoch) ===")
    pipe, h = run(splits, epochs)
    best = max(h["val_acc"])
    log("epoch: " + " | ".join(f"{a:.4f}" for a in h["val_acc"]))
    log(f"Val accuracy tốt nhất {best:.4f} (Lab 04 yêu cầu ≥ 0,97) | huấn luyện {pipe.model.fit_time_:.1f} s")
    return dict(val_acc=h["val_acc"], best_val_acc=best, fit_s=pipe.model.fit_time_)


def run_e1(splits, log, epochs, lrs, momenta):
    log(f"\n=== E1. Learning rate γ × momentum α ({epochs} epoch); γ_eff = γ / (1 − α) ===")
    acc = np.full((len(momenta), len(lrs)), np.nan)
    for i, a in enumerate(momenta):
        for j, lr in enumerate(lrs):
            _, h = run(splits, epochs, lr=lr, momentum=a)
            acc[i, j] = last_val_acc(h)
            log(f"α = {a:<4} γ = {lr:<6} γ_eff = {lr / (1 - a):<6.3g}: "
                + ("phân kỳ" if np.isnan(acc[i, j]) else f"val acc {acc[i, j]:.4f}"))
    i, j = np.unravel_index(np.nanargmax(acc), acc.shape)
    best_lr, best_mom = lrs[j], momenta[i]
    log(f"Chọn γ = {best_lr}, α = {best_mom} (val acc {acc[i, j]:.4f})")
    return dict(lrs=list(lrs), momenta=list(momenta), val_acc=acc, best_lr=best_lr, best_momentum=best_mom,
                epochs=epochs)


def run_e2(splits, log, epochs, batch_sizes):
    log(f"\n=== E2. Kích thước batch ({epochs} epoch, γ = {BASE['lr']}, α = {BASE['momentum']}) ===")
    n = len(splits["train"][1])
    rows = []
    log(f"{'batch':>6} | {'số cập nhật':>11} | {'s/epoch':>7} | {'val acc':>7}")
    for bs in batch_sizes:
        pipe, h = run(splits, epochs, batch_size=bs)
        r = dict(batch_size=bs, updates=int(np.ceil(n / bs)) * epochs, s_per_epoch=pipe.model.fit_time_ / len(h["time"]),
                 val_acc=last_val_acc(h))
        rows.append(r)
        log(f"{bs:>6} | {r['updates']:>11} | {r['s_per_epoch']:7.2f} | {r['val_acc']:7.4f}")
    best = max(rows, key=lambda r: np.nan_to_num(r["val_acc"], nan=-1))["batch_size"]
    log(f"Chọn batch = {best}")
    return dict(rows=rows, best=best, epochs=epochs, lr=BASE["lr"], momentum=BASE["momentum"])


def run_e3(splits, log, epochs, archs):
    log(f"\n=== E3. Độ rộng và độ sâu ({epochs} epoch) ===")
    rows = []
    log(f"{'kiến trúc':>26} | {'tham số':>9} | {'train acc':>9} | {'val acc':>7}")
    for hidden in archs:
        pipe, h = run(splits, epochs, hidden=hidden)
        name = "784-" + "-".join(map(str, hidden)) + "-10"
        r = dict(hidden=list(hidden), name=name, n_params=pipe.model.n_params_, train_acc=h["train_acc"][-1],
                 val_acc=last_val_acc(h))
        rows.append(r)
        log(f"{name:>26} | {r['n_params']:>9,} | {r['train_acc']:9.4f} | {r['val_acc']:7.4f}".replace(",", "."))
    best = tuple(max(rows, key=lambda r: np.nan_to_num(r["val_acc"], nan=-1))["hidden"])
    log(f"Chọn kiến trúc hidden = {best}")
    return dict(rows=rows, best=list(best), epochs=epochs)


def run_e4(splits, Z512, y512, log, epochs):
    log(f"\n=== E4. Hàm kích hoạt và vanishing gradient (mạng 784 → 128 × 6 → 10, {epochs} epoch) ===")
    deep = (128,) * 6
    setups = {"sigmoid": "xavier", "tanh": "xavier", "relu": "he"}
    grad_norms, act_acc = {}, {}
    for act, init in setups.items():
        layers = build_mlp([784, *deep, 10], act, init=init, seed=SEED)
        logits, cache = forward_with_cache(layers, Z512, train=False)
        _, delta = softmax_cross_entropy(logits, y512)
        backward(cache, delta)
        grad_norms[act] = [float(np.linalg.norm(l.dW)) for l in linears(layers)]
        _, h = run(splits, epochs, hidden=deep, activation=act, init=init)
        act_acc[act] = last_val_acc(h)
        gn = grad_norms[act]
        log(f"{act:>7}: ‖∂J/∂W‖ lớp 1 = {gn[0]:.2e} | lớp cuối = {gn[-1]:.2e} | lớp cuối / lớp 1 = "
            f"{gn[-1] / gn[0]:.1f} | val acc = {act_acc[act]:.4f}")
    return dict(grad_norms=grad_norms, val_acc=act_acc)


def run_e5(splits, Z1000, log, epochs):
    log("\n=== E5. Khởi tạo trọng số ===")
    sym = {}
    for init in ("const", "xavier"):
        pipe, h = run(splits, epochs, hidden=(64,), activation="tanh", init=init)
        W1 = linears(pipe.model.layers_)[0].W
        sym[init] = dict(distinct_rows=int(np.unique(W1, axis=0).shape[0]), val_acc=last_val_acc(h))
        log(f"(a) {init:>7}: số nơ-ron ẩn khác nhau = {sym[init]['distinct_rows']:2d} / 64 sau {epochs} epoch | "
            f"val acc = {sym[init]['val_acc']:.4f}")
    stds = {}
    for init in ("xavier", "he"):
        _, cache = forward_with_cache(build_mlp([784] + [256] * 10 + [10], init=init, seed=SEED), Z1000, train=False)
        stds[init] = [float(a_out.std()) for layer, _, a_out, _ in cache
                      if getattr(layer, "name", None) == "relu"]
        log(f"(b) {init:>7}: std đầu ra ReLU lớp 1 = {stds[init][0]:.3f} | lớp 10 = {stds[init][-1]:.4f}")
    return dict(symmetry=sym, relu_std=stds)


def run_e6(splits, log, epochs):
    log(f"\n=== E6. Điều chuẩn L2 và dropout ({epochs} epoch) ===")
    hist = {name: run(splits, epochs, **ch)[1] for name, ch in REGS.items()}
    log(f"{'cấu hình':>18} | {'train acc':>9} | {'val acc':>7} | {'val loss':>8} | {'val loss nhỏ nhất':>17}")
    for name, h in hist.items():
        log(f"{name:>18} | {h['train_acc'][-1]:9.4f} | {h['val_acc'][-1]:7.4f} | {h['val_loss'][-1]:8.4f} | "
            f"{min(h['val_loss']):17.4f}")
    best = max(hist, key=lambda k: hist[k]["val_acc"][-1])
    log(f"Chọn điều chuẩn: {best}")
    return dict(rows={k: dict(train_acc=h["train_acc"][-1], val_acc=h["val_acc"][-1], val_loss=h["val_loss"][-1],
                              min_val_loss=min(h["val_loss"])) for k, h in hist.items()}, best=best,
                histories={k: dict(val_loss=h["val_loss"], val_acc=h["val_acc"]) for k, h in hist.items()},
                epochs=epochs)


def run_e7(splits, log, candidates, max_epochs, patience):
    """E1–E6 đổi TỪNG yếu tố so với baseline, nên ghép mọi lựa chọn tốt nhất chưa chắc tốt nhất (các yếu tố
    tương tác). So ba ứng viên với early stopping theo val loss; luật chọn: val accuracy cao nhất ở epoch
    được khôi phục (giống Multinomial NB)."""
    log(f"\n=== E7. Ba ứng viên, tối đa {max_epochs} epoch, early stopping patience {patience} ===")
    pipes, rows = {}, {}
    log(f"{'ứng viên':>24} | {'epoch tốt nhất':>14} | {'val acc':>7} | {'val loss':>8} | {'thời gian':>9}")
    for name, cfg in candidates.items():
        pipe = make_mlp(cfg, max_epochs, patience)
        t0 = time.perf_counter()
        pipe.fit(*splits["train"], *splits["val"])
        wall = time.perf_counter() - t0
        m = pipe.model
        k = m.best_epoch_ - 1
        # Định nghĩa chung: gồm tiền xử lý, cập nhật và val loss; không gồm loss/acc trên train mỗi epoch
        rows[name] = dict(config=cfg_json(cfg), best_epoch=m.best_epoch_, stopped_epoch=m.n_epochs_,
                          val_acc=m.history_["val_acc"][k], val_loss=m.history_["val_loss"][k],
                          fit_time=wall - m.diag_time_)
        pipes[name] = pipe
        r = rows[name]
        log(f"{name:>24} | {r['best_epoch']:>14} | {r['val_acc']:7.4f} | {r['val_loss']:8.4f} | {r['fit_time']:8.1f}s")
    chosen = max(rows, key=lambda n: rows[n]["val_acc"])
    final = pipes[chosen]
    log(f"Chọn {chosen}: {json.dumps(rows[chosen]['config'], ensure_ascii=False)}")
    log(f"Dừng ở epoch {final.model.n_epochs_}, khôi phục epoch {final.model.best_epoch_}")
    return final, chosen, rows


def run_e8(splits, final, chosen, rows, log, perm_repeats, n_jobs, quick, perm_n=None, summary_extra=None):
    """Đánh giá cuối (test dùng một lần), cùng các phân tích chung với RF, LR, NB, CNN."""
    log("\n=== E8. Đánh giá cuối (test dùng một lần) ===")
    m = final.model
    res = evaluate(final, splits, fit=False)
    res["fit_time"] = rows[chosen]["fit_time"]
    log(f"Huấn luyện: {res['fit_time']:.1f} s ({n_jobs} luồng BLAS) | {m.n_params_:,} tham số".replace(",", "."))
    for name in ("train", "val", "test"):
        s = res[name]
        log(f"{name:>5}: acc {s['accuracy']:.4f} | P {s['precision_macro']:.4f} | R {s['recall_macro']:.4f} | "
            f"F1 {s['f1_macro']:.4f} | dự đoán {s['predict_time']:.2f} s")
    for name in ("train", "val", "test"):
        log(format_confusion(res[name]["confusion_matrix"], f"\n[{name}] ma trận nhầm lẫn (hàng: nhãn thật, cột: dự đoán)"))

    out = {}
    nc = NearestCentroidNP().fit(*splits["train"])
    prob, P_all = {}, {}
    for name in ("train", "val", "test"):
        X, y = splits[name]
        P = P_all[name] = final.predict_proba(X)
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
            f"lỗi theo cặp MLP {o['pairs']}")
        log(f"[{name}]     5 cặp nhầm nhiều nhất: {o['top_pairs']}")
        n_nc = (1 - o["nearest_centroid"]["accuracy"]) * len(splits[name][1])
        red = {k: 1 - o["pairs"][k] / max(o["nearest_centroid"]["pairs"][k], 1) for k in o["pairs"]}
        log(f"[{name}] So với tâm gần nhất: acc {o['nearest_centroid']['accuracy']:.4f}; MLP giảm lỗi "
            f"{1 - o['n_errors'] / n_nc:.1%} tổng, theo cặp " + ", ".join(f"{k}: {v:.1%}" for k, v in red.items()))

    # Giải thích W^(1): W = (khởi tạo, bị L2 co lại) + (phần học từ dữ liệu). Ở dead pixel mọi ảnh cùng giá trị,
    # nên ΔW ở đó chỉ là L2 co W_0 (cộng hằng số giống bias): ΔW ≈ −k·W_0. Ước lượng k bằng bình phương tối
    # thiểu trên dead pixel; phần học từ dữ liệu R = ΔW + k·W_0.
    X_tr, y_tr = splits["train"]
    W1 = linears(m.layers_)[0].W.astype(np.float64)
    W0 = linears(m.initial_layers())[0].W.astype(np.float64)
    dead = X_tr.std(axis=0) == 0
    dW = W1 - W0
    k_shrink = -(dW[:, dead] * W0[:, dead]).sum() / (W0[:, dead] ** 2).sum() if dead.any() else 0.0
    R = dW + k_shrink * W0
    corr = float(np.corrcoef(dW[:, dead].ravel(), W0[:, dead].ravel())[0, 1]) if dead.any() else float("nan")
    log(f"\nW^(1): {dead.sum()} dead pixel | tương quan(ΔW, W_0) trên dead pixel = {corr:.3f} | L2 co phần khởi tạo "
        f"còn {1 - k_shrink:.0%} | ‖ΔW‖ = {np.linalg.norm(dW):.1f}, phần học từ dữ liệu ‖R‖ = {np.linalg.norm(R):.1f}")
    top = np.argsort(-np.linalg.norm(R, axis=1))[:16]                # 16 nơ-ron có phần học từ dữ liệu mạnh nhất

    # Độ quan trọng: hoán vị khối 4×4 (chung cho mọi mô hình); so với Fisher (H4) làm một lần ở compare_models.
    X_va, y_va = splits["val"]
    n = perm_n or len(X_va)
    t0 = time.perf_counter()
    drops = block_permutation_importance(final, X_va[:n], y_va[:n], n_repeats=perm_repeats, seed=SEED)
    log(f"Hoán vị khối 4×4 trên val ({time.perf_counter() - t0:.0f} s, {n} ảnh): khối quan trọng nhất làm giảm "
        f"acc {drops.max():.4f}")
    out["w1"] = dict(dead_pixels=int(dead.sum()), corr_dw_w0_dead=corr, shrink_k=float(k_shrink),
                     norm_dw=float(np.linalg.norm(dW)), norm_r=float(np.linalg.norm(R)))
    out["w1_top"] = dict(idx=top, R=R[top], n_hidden=W1.shape[0])

    out["confident_errors"] = confident_error_records(splits["test"][1], P_all["test"])

    # epochs: số epoch của trọng số được giữ; dùng khi huấn luyện lại mà không có val (sklearn_reference)
    cfg = dict(rows[chosen]["config"], seed=SEED, max_epochs=m.max_epochs, patience=m.patience,
               epochs=m.best_epoch_, features="make_features('none', 'scalar'): pixel [0, 1], 784, chuẩn hoá vô hướng")
    save_final_metrics(
        KEY, "MLP", res, cfg, probabilistic=prob,
        block_importance=drops, n_jobs=n_jobs,
        extra=dict(best_epoch=m.best_epoch_, stopped_epoch=m.n_epochs_, n_params=m.n_params_, chosen_candidate=chosen,
                   candidates=rows, selection_rule="val accuracy cao nhất ở mỗi thí nghiệm và giữa các ứng viên cuối",
                   fit_time_with_train_diagnostics=rows[chosen]["fit_time"] + m.diag_time_,
                   w1=out["w1"], **(summary_extra or {})),
        quick=quick)
    return out


# ---------------------------------------------------------------- chương trình chính
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="train 10.000 ảnh, lưới nhỏ, ít epoch: chỉ để kiểm tra code")
    ap.add_argument("--blas-threads", type=int, default=8, help="số luồng BLAS (mặc định 8, xem docstring)")
    args = ap.parse_args()
    with threadpool_limits(args.blas_threads, user_api="blas"):
        experiment(args.quick, args.blas_threads)


def experiment(q, n_threads):
    log = Logger(OUT_DIR / "log.txt")
    t0 = time.perf_counter()
    splits = load_splits()
    if q:
        splits = subset_train(splits, 10_000, SEED)
    log(f"train {splits['train'][0].shape} | val {splits['val'][0].shape} | test {splits['test'][0].shape}"
        f" | {n_threads} luồng BLAS | quick = {q}")
    ep_sweep, ep_reg = (1, 2) if q else (5, 15)

    feats = make_features("none", "scalar")
    Z_tr = feats.fit_transform(splits["train"][0]).astype(np.float32)
    summary = {"base": cfg_json(BASE)}
    summary["E0"] = run_e0(splits, log, 2 if q else 10)
    e1 = summary["E1"] = run_e1(splits, log, ep_sweep, (0.05, 0.5) if q else LRS, (0.5, 0.9) if q else MOMENTA)
    e2 = summary["E2"] = run_e2(splits, log, ep_sweep, (128, 256) if q else BATCH_SIZES)
    e3 = summary["E3"] = run_e3(splits, log, ep_sweep, ((64,), (256, 128)) if q else ARCHS)
    summary["E4"] = run_e4(splits, Z_tr[:512], splits["train"][1][:512], log, ep_sweep)
    summary["E5"] = run_e5(splits, Z_tr[:1000], log, ep_sweep if q else 3)
    e6 = summary["E6"] = run_e6(splits, log, ep_reg)

    best_reg = REGS[e6["best"]]
    candidates = {
        "A: gộp E1+E2+E3+E6": dict(BASE, hidden=tuple(e3["best"]), lr=e1["best_lr"], momentum=e1["best_momentum"],
                                   batch_size=e2["best"], **best_reg),
        "B: baseline + E3 + E6": dict(BASE, hidden=tuple(e3["best"]), **best_reg),
        "C: baseline + E6": dict(BASE, **best_reg),
    }
    final, chosen, rows = run_e7(splits, log, candidates, 3 if q else FINAL_EPOCHS, 1 if q else FINAL_PATIENCE)
    summary["E7"] = dict(candidates=rows, chosen=chosen, history=final.model.history_,
                         best_epoch=final.model.best_epoch_)
    summary["final"] = run_e8(
        splits, final, chosen, rows, log, perm_repeats=1 if q else PERM_REPEATS, n_jobs=n_threads,
        quick=q, perm_n=2000 if q else None,
        summary_extra=dict(e1_lr_momentum=dict(lrs=e1["lrs"], momenta=e1["momenta"], val_acc=e1["val_acc"]),
                           e2_batch=e2["rows"], e3_arch=e3["rows"], e6_reg=e6["rows"]))
    summary["total_s"] = time.perf_counter() - t0
    log(f"\nTổng thời gian: {summary['total_s'] / 60:.1f} phút")
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'summary.json')}, log.txt và figures/")


if __name__ == "__main__":
    main()
