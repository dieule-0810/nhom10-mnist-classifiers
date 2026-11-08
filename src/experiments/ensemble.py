"""Bước 4 + 5 của slide khuyến nghị bài 06: đo diversity rồi gộp mô hình (soft voting, seed ensemble CNN).

Không dò siêu tham số mới: dựng lại mô hình cuối của từng người từ "config" trong
results/<key>/final_metrics.json, bằng đúng lớp trong src/models/ mà script của mô hình đó dùng.
CNN được huấn luyện với nhiều seed; seed chỉ đổi khởi tạo và thứ tự mini-batch. Mỗi CNN của seed ensemble
được lưu thành results/ensemble/cnn_s<seed>.pkl, dùng lại ở src/experiments/final_models.py.

Chạy từ gốc repo, SAU khi đã có kết quả các mô hình (cần mã nguồn của chúng trong repo):
    python -m src.experiments.ensemble --quick       # kiểm tra code (train 10.000 ảnh, 2 seed CNN, 1 epoch)
    python -m src.experiments.ensemble               # bản đầy đủ: 5 seed CNN (khoảng 30 phút trên CPU)

Mọi lựa chọn (tổ hợp, trọng số) làm trên val; test chỉ dùng ở E5 (bẫy số 3 của slide 20).

    E1  Diversity: với mỗi cặp, "% cùng sai" = số ảnh cả hai cùng sai / số ảnh ít nhất một mô hình sai
    E2  Soft voting các mô hình cổ điển, so với mô hình tốt nhất trong tổ hợp
    E3  Seed ensemble CNN: accuracy theo số mạng k = 1..K, soft và hard voting
    E4  Trộn CNN ensemble với mô hình cổ điển: trọng số của CNN chọn trên val
    E5  Bảng tổng kết trên test (một lần) và những ảnh ensemble vẫn sai

Kết quả ghi vào results/ensemble/: log.txt, summary.json, cnn_s*.pkl và final_metrics.json (schema chung) của
phương án E5 chọn trên val, để compare_models xếp ensemble cạnh các mô hình đơn. fit_time là tổng thời gian huấn
luyện các thành viên (chạy lần lượt); không có block_importance (hoán vị khối với 5 CNN quá đắt).
Hình (div_pairs, ens_seeds, ens_weights, eval_confusion, eval_residual_errors) do src/plots/ensemble.py vẽ từ
summary.json và final_metrics.json (gọi ở cuối script; vẽ lại riêng: python -m src.plots.ensemble).

Tham khảo:
    [1] L. I. Kuncheva, C. J. Whitaker, "Measures of diversity in classifier ensembles," Machine Learning,
        vol. 51, pp. 181–207, 2003.
    [2] B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty estimation
        using deep ensembles," NeurIPS, 2017.
"""

import argparse
import itertools
import os
import time

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import DISPLAY_NAMES, ENSEMBLE_KEY, SEED, SPLITS
from src.data.features import TreeFeatures, make_features
from src.data.preprocess import load_splits, subset_train
from src.evaluation.analysis import confident_error_records, log_loss, probabilistic_scores
from src.evaluation.metrics import classification_scores
from src.evaluation.results_io import (Logger, load_final_metrics, model_dir, rel_path, save_final_metrics,
                                       save_model, write_json)
from src.models.pipeline import Pipeline
from src.plots.ensemble import plot_all

KEY = ENSEMBLE_KEY
OUT_DIR = model_dir(KEY)
CLASSICAL = ("random_forest", "logistic_regression", "naive_bayes", "mlp")
SHORT = {"random_forest": "RF", "logistic_regression": "LR", "naive_bayes": "NB", "mlp": "MLP"}
WEIGHTS = (1, 2, 3, 5, 8)


# ---------------------------------------------------------------- dựng mô hình từ final_metrics.json
def build_random_forest(cfg, n_jobs):
    from src.models.random_forest import make_random_forest

    model = make_random_forest(n_estimators=cfg["n_estimators"], criterion=cfg["criterion"],
                               max_features=cfg["max_features"], max_depth=cfg["max_depth"],
                               min_samples_leaf=cfg["min_samples_leaf"], n_jobs=n_jobs, seed=SEED)
    return Pipeline(TreeFeatures(mask=cfg["mask"], n_bins=cfg["n_bins"]), model)


def build_logistic_regression(cfg, n_jobs):
    from src.models.logistic_regression import SoftmaxRegressionNP

    feats = make_features(cfg["mask"], cfg.get("scale", "scalar"))
    model = SoftmaxRegressionNP(eta=cfg["eta"], lam=cfg["lam"], max_epochs=cfg["max_epochs"],
                                batch_size=cfg["batch_size"], momentum=cfg["momentum"], patience=cfg["patience"],
                                seed=SEED, tol=None)
    return Pipeline(feats, model)


def build_naive_bayes(cfg, n_jobs):
    from src.models.naive_bayes import MultinomialNBNP

    return Pipeline(make_features(cfg["mask"], "none"), MultinomialNBNP(alpha=cfg["alpha"]))


def build_mlp(cfg, n_jobs):
    from src.experiments.mlp import make_mlp

    return make_mlp(cfg, cfg["max_epochs"], cfg["patience"])


BUILDERS = {
    "random_forest": build_random_forest,
    "logistic_regression": build_logistic_regression,
    "naive_bayes": build_naive_bayes,
    "mlp": build_mlp,
}


def fit_proba(pipe, splits):
    """Huấn luyện trên train (val cho mô hình có dừng sớm).

    Trả về (xác suất trên mọi tập, thời gian fit, {tập: thời gian predict_proba}); pipe đã huấn luyện.
    """
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr, X_va, y_va)
    fit_s = time.perf_counter() - t0
    P, pred_s = {}, {}
    for s in SPLITS:
        t0 = time.perf_counter()
        P[s] = pipe.predict_proba(splits[s][0])
        pred_s[s] = time.perf_counter() - t0
    return P, fit_s, pred_s


# ---------------------------------------------------------------- độ đo gộp
def acc(P, y):
    return float((np.asarray(P).argmax(axis=1) == y).mean())


def soft_vote(Ps, weights=None):
    """Trung bình (có trọng số) xác suất."""
    w = np.ones(len(Ps)) if weights is None else np.asarray(weights, dtype=float)
    return np.tensordot(w / w.sum(), np.asarray(Ps), axes=1)


def hard_vote(Ps):
    """Đa số phiếu nhãn; hoà thì chọn lớp có tổng xác suất lớn hơn (không thiên về nhãn nhỏ)."""
    votes = np.zeros_like(Ps[0])
    for P in Ps:
        votes[np.arange(len(P)), P.argmax(axis=1)] += 1
    return votes + 1e-6 * soft_vote(Ps)


def co_error(Pa, Pb, y):
    """(cả hai cùng sai, ít nhất một sai, % cùng sai) như slide 7."""
    wa, wb = Pa.argmax(1) != y, Pb.argmax(1) != y
    both, either = int((wa & wb).sum()), int((wa | wb).sum())
    return both, either, both / either if either else float("nan")


# ---------------------------------------------------------------- các thí nghiệm
def run_e1(members, y_va, log):
    log("\n=== E1. Diversity trên val: % cùng sai = cả hai sai / ít nhất một sai ===")
    names = list(members)
    n = len(names)
    M = np.full((n, n), np.nan)
    rows = []
    log(f"{'cặp':<28}{'cả hai sai':>11}{'≥ 1 sai':>9}{'% cùng sai':>12}")
    for i, j in itertools.combinations(range(n), 2):
        both, either, r = co_error(members[names[i]]["val"], members[names[j]]["val"], y_va)
        M[i, j] = M[j, i] = r
        rows.append(dict(a=names[i], b=names[j], both=both, either=either, co_error=r))
        log(f"{names[i] + ' vs ' + names[j]:<28}{both:>11}{either:>9}{r:>12.1%}")
    cnn = [k for k in names if k.startswith("CNN")]
    out = dict(pairs=rows, names=names, matrix=M)
    if len(cnn) >= 3:
        wrong = np.array([members[k]["val"].argmax(1) != y_va for k in cnn[:3]])
        all3, any3 = int(wrong.all(0).sum()), int(wrong.any(0).sum())
        out["cnn3"] = dict(all_wrong=all3, any_wrong=any3, share=all3 / any3)
        log(f"3 CNN ({', '.join(cnn[:3])}): {any3} ảnh có ít nhất 1 mạng sai, {all3} ảnh ({all3 / any3:.1%}) cả ba "
            f"cùng sai (slide 11: 20,0%)")
    log("Quy tắc nhanh (slide 7): % cùng sai > 60% thì gộp gần như không thêm thông tin.")
    return out


def run_e2(members, classical, y_va, log):
    log("\n=== E2. Soft voting các mô hình cổ điển (val) ===")
    rows = []
    for r in range(2, len(classical) + 1):
        for combo in itertools.combinations(classical, r):
            a = acc(soft_vote([members[k]["val"] for k in combo]), y_va)
            best = max(combo, key=lambda k: acc(members[k]["val"], y_va))
            b = acc(members[best]["val"], y_va)
            rows.append(dict(combo=list(combo), acc=a, best_member=best, best_member_acc=b, gain=a - b))
            log(f"{' + '.join(combo):<20}: {a:.4f} | tốt nhất trong nhóm {best} {b:.4f} | chênh {a - b:+.4f}")
    best = max(rows, key=lambda r: r["acc"]) if rows else None
    if best:
        log(f"Tổ hợp cổ điển tốt nhất trên val: {' + '.join(best['combo'])} ({best['acc']:.4f})")
    return dict(rows=rows, best=best)


def run_e3(members, cnn, y_va, log):
    log(f"\n=== E3. Seed ensemble CNN trên val: k = 1..{len(cnn)} mạng ===")
    singles = [acc(members[k]["val"], y_va) for k in cnn]
    log("Từng mạng: " + " | ".join(f"{k} {a:.4f}" for k, a in zip(cnn, singles)) +
        f" | trung bình {np.mean(singles):.4f}")
    rows = []
    log(f"{'k':>3}{'soft (k đầu)':>14}{'soft TB mọi tổ hợp':>20}{'min':>8}{'max':>8}{'hard (k đầu)':>14}")
    for k in range(1, len(cnn) + 1):
        subs = [acc(soft_vote([members[c]["val"] for c in s]), y_va) for s in itertools.combinations(cnn, k)]
        first = [members[c]["val"] for c in cnn[:k]]
        row = dict(k=k, soft_first=acc(soft_vote(first), y_va), soft_mean=float(np.mean(subs)),
                   soft_min=float(np.min(subs)), soft_max=float(np.max(subs)), hard_first=acc(hard_vote(first), y_va))
        rows.append(row)
        log(f"{k:>3}{row['soft_first']:>14.4f}{row['soft_mean']:>20.4f}{row['soft_min']:>8.4f}{row['soft_max']:>8.4f}"
            f"{row['hard_first']:>14.4f}")
    full = rows[-1]
    log(f"Ensemble {len(cnn)} mạng: {full['soft_first'] - max(singles):+.4f} so với mạng tốt nhất, "
        f"{full['soft_first'] - np.mean(singles):+.4f} so với trung bình từng mạng")
    log("Hard voting chỉ đếm phiếu, bỏ mất độ tự tin: một mạng rất chắc đúng thua hai mạng hơi nghiêng về nhãn sai.")
    return dict(singles=dict(zip(cnn, singles)), rows=rows)


def run_e4(members, cnn, classical_sets, y_va, log):
    log(f"\n=== E4. CNN ensemble ({len(cnn)} mạng) trộn với mô hình cổ điển: trọng số w của CNN chọn trên val ===")
    P_cnn = soft_vote([members[k]["val"] for k in cnn])
    log(f"Chỉ CNN ensemble: {acc(P_cnn, y_va):.4f}")
    rows = []
    for combo in classical_sets:
        Ps = [members[k]["val"] for k in combo]
        accs = [acc(soft_vote([P_cnn, *Ps], [w, *[1] * len(Ps)]), y_va) for w in WEIGHTS]
        w_best = WEIGHTS[int(np.argmax(accs))]
        rows.append(dict(combo=list(combo), accs=dict(zip(WEIGHTS, accs)), w_best=w_best, acc_best=max(accs)))
        log(f"CNN-ens + {' + '.join(combo):<14}: " + " | ".join(f"w={w}: {a:.4f}" for w, a in zip(WEIGHTS, accs))
            + f"  -> chọn w = {w_best}")
    best = max(rows, key=lambda r: r["acc_best"]) if rows else None
    if best:
        log(f"Tốt nhất trên val: CNN-ens (w = {best['w_best']}) + {' + '.join(best['combo'])}: {best['acc_best']:.4f}")
    return dict(rows=rows, best=best, cnn_only=acc(P_cnn, y_va))


def run_e5(members, cnn, classical, e2, e4, splits, log):
    log("\n=== E5. Bảng tổng kết trên test (cấu hình đã chọn trên val; test dùng một lần) ===")
    y_te = splits["test"][1]
    y_va = splits["val"][1]
    cands = [(k, [k], None) for k in classical]
    if cnn:
        cands.append((f"CNN đơn ({cnn[0]})", [cnn[0]], None))
    if e2["best"]:
        cands.append(("Soft voting: " + " + ".join(SHORT.get(k, k) for k in e2["best"]["combo"]), e2["best"]["combo"], None))
    if len(cnn) > 1:
        cands.append((f"CNN ensemble {len(cnn)} seed", cnn, None))
    if e4["best"]:
        combo = e4["best"]["combo"]
        tag = " + ".join(SHORT.get(k, k) for k in combo)
        cands.append((f"{tag} + CNN ens, cùng trọng số", ["CNN_ENS", *combo], [1] * (len(combo) + 1)))
        cands.append((f"{tag} + CNN ens, CNN × {e4['best']['w_best']}", ["CNN_ENS", *combo],
                      [e4["best"]["w_best"], *[1] * len(combo)]))

    def proba(keys, weights, split):
        Ps = [soft_vote([members[c][split] for c in cnn]) if k == "CNN_ENS" else members[k][split] for k in keys]
        return soft_vote(Ps, weights)

    rows = []
    log(f"{'phương án':<42}{'val acc':>9}{'test acc':>10}{'ảnh sai':>9}{'log-loss':>10}")
    for name, keys, w in cands:
        Pv, Pt = proba(keys, w, "val"), proba(keys, w, "test")
        a, n_err = acc(Pt, y_te), int((Pt.argmax(1) != y_te).sum())
        rows.append(dict(name=name, members=keys, weights=w, val_acc=acc(Pv, y_va), test_acc=a, test_errors=n_err,
                         test_log_loss=log_loss(y_te, Pt)))
        log(f"{name:<42}{acc(Pv, y_va):>9.4f}{a:>10.4f}{n_err:>9}{log_loss(y_te, Pt):>10.4f}")
    final = max(rows, key=lambda r: r["val_acc"])
    log(f"Chọn theo val: {final['name']} -> test {final['test_acc']:.4f} ({final['test_errors']} ảnh sai)")
    final["proba"] = {s: proba(final["members"], final["weights"], s) for s in SPLITS}
    final["confident_errors"] = confident_error_records(y_te, final["proba"]["test"])
    return dict(rows=rows, final=final)


def save_final(final, cnn, fit_times, pred_times, splits, blas_threads, rf_n_jobs, quick, log):
    """final_metrics.json của phương án chọn trên val, cùng schema với các mô hình đơn."""
    used = [m for k in final["members"] for m in (cnn if k == "CNN_ENS" else [k])]
    res = {"fit_time": sum(fit_times[m] for m in used)}
    for s in SPLITS:
        res[s] = classification_scores(splits[s][1], final["proba"][s].argmax(axis=1))
        res[s]["predict_time"] = sum(pred_times[m][s] for m in used)
    prob = {s: probabilistic_scores(splits[s][1], final["proba"][s]) for s in SPLITS}
    config = dict(name=final["name"], members=final["members"], weights=final["weights"], cnn_seeds=cnn,
                  voting="soft", chosen_on="val")
    # runtime.n_jobs: luồng BLAS của các CNN (gần hết thời gian huấn luyện); RF chạy song song bằng joblib riêng
    rf_threads = os.cpu_count() if rf_n_jobs in (-1, None) else rf_n_jobs
    extra = dict(rf_n_jobs=rf_threads) if "random_forest" in used else {}
    path = save_final_metrics(KEY, DISPLAY_NAMES[KEY], res, config, probabilistic=prob, n_jobs=blas_threads, quick=quick,
                              extra=dict(extra, member_fit_times={m: fit_times[m] for m in used},
                                         fit_time_note="tổng thời gian huấn luyện các thành viên, chạy lần lượt"))
    log(f"Đã ghi {rel_path(path)} ({final['name']}; fit {res['fit_time']:.0f} s cộng dồn {len(used)} thành viên)")


# ---------------------------------------------------------------- chương trình chính
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="train 10.000 ảnh, 2 seed CNN, 1 epoch: chỉ để kiểm tra code")
    ap.add_argument("--seeds", type=int, default=5, help="số seed CNN (mặc định 5)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="số luồng cho Random Forest")
    ap.add_argument("--blas-threads", type=int, default=8, help="số luồng BLAS cho MLP, CNN (xem src/experiments/mlp.py)")
    args = ap.parse_args()
    with threadpool_limits(args.blas_threads, user_api="blas"):
        experiment(args)


def experiment(args):
    q = args.quick
    log = Logger(OUT_DIR / "log.txt")
    t0 = time.perf_counter()
    splits = load_splits()
    if q:
        splits = subset_train(splits, 10_000, SEED)
    y_va = splits["val"][1]
    log(f"train {splits['train'][0].shape} | val {splits['val'][0].shape} | test {splits['test'][0].shape} | quick = {q}")

    log("\n=== Dựng lại mô hình cuối từ results/<key>/final_metrics.json ===")
    members, fit_times, pred_times, recorded = {}, {}, {}, {}
    for key in CLASSICAL:
        fm = load_final_metrics(key)
        if fm is None or key not in BUILDERS:
            log(f"{DISPLAY_NAMES[key]:<20}: bỏ qua ({'chưa có kết quả' if fm is None else 'chưa có build_' + key})")
            continue
        try:
            pipe = BUILDERS[key](fm["config"], args.n_jobs)
        except ImportError as e:
            log(f"{DISPLAY_NAMES[key]:<20}: bỏ qua (thiếu mã nguồn: {e})")
            continue
        members[key], fit_times[key], pred_times[key] = fit_proba(pipe, splits)
        recorded[key] = fm["splits"]["val"]["accuracy"]
        log(f"{DISPLAY_NAMES[key]:<20}: val acc {acc(members[key]['val'], y_va):.4f} "
            f"(final_metrics.json: {recorded[key]:.4f}) | {fit_times[key]:.0f} s")
    classical = list(members)

    fm_cnn = load_final_metrics("cnn")
    cnn = []
    if fm_cnn is None:
        log("CNN: chưa có results/cnn/final_metrics.json, bỏ qua")
    else:
        from src.experiments.cnn import make_cnn

        cfg = dict(fm_cnn["config"], channels=tuple(fm_cnn["config"]["channels"]))
        if q:
            cfg["max_epochs"] = 1
        for s in range(2 if q else args.seeds):
            name = f"CNN-s{s}"
            pipe = make_cnn(cfg, seed=s)
            members[name], fit_times[name], pred_times[name] = fit_proba(pipe, splits)
            if not q:
                save_model(KEY, pipe, f"cnn_s{s}")
            cnn.append(name)
            log(f"{name:<20}: val acc {acc(members[name]['val'], y_va):.4f} | {fit_times[name]:.0f} s")
    if len(members) < 2:
        log("Cần ít nhất 2 mô hình để gộp.")
        return

    summary = dict(fit_times=fit_times, recorded_val_acc=recorded, quick=q)
    summary["E1"] = run_e1(members, y_va, log)
    summary["E2"] = run_e2(members, classical, y_va, log)
    summary["E3"] = run_e3(members, cnn, y_va, log) if len(cnn) > 1 else None
    # E4 thử CNN-ens + từng mô hình cổ điển, và + tổ hợp cổ điển tốt nhất của E2
    sets = [[k] for k in classical]
    if summary["E2"]["best"]:
        sets.append(summary["E2"]["best"]["combo"])
    summary["E4"] = run_e4(members, cnn, sets, y_va, log) if cnn and classical else dict(best=None)
    summary["E5"] = run_e5(members, cnn, classical, summary["E2"], summary["E4"], splits, log)
    save_final(summary["E5"]["final"], cnn, fit_times, pred_times, splits, args.blas_threads, args.n_jobs, q, log)
    del summary["E5"]["final"]["proba"]
    summary["total_s"] = time.perf_counter() - t0
    log(f"\nTổng thời gian: {summary['total_s'] / 60:.1f} phút")
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'summary.json')}, log.txt và figures/")


if __name__ == "__main__":
    main()
