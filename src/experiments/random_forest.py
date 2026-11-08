"""Thí nghiệm Random Forest trên MNIST, nối từ EDA (eda/notebooks/eda.ipynb) tới đánh giá cuối.

Chạy từ gốc repo:
    python -m src.experiments.random_forest --quick        # kiểm tra code chạy được (khoảng 1 phút)
    python -m src.experiments.random_forest                # bản đầy đủ, mọi lõi (--n-jobs để giới hạn)

Quy tắc chung: mỗi thí nghiệm chỉ đổi một yếu tố so với cấu hình hiện tại, chọn theo
validation, tập test chỉ dùng một lần ở E6. Mọi tham số tiền xử lý học từ splits["train"].

    E0  Mốc: tâm gần nhất, một cây CART đầy đủ
    E1  Số ngưỡng ứng viên mỗi pixel (n_bins)            EDA 5.3: "chỉ cần ít ngưỡng"
    E2  Bỏ pixel: none / dead / rare / crop24 / crop20  EDA 1.3
    E3  Số cây T: OOB và validation theo T, soft / hard  thí nghiệm siêu tham số có biểu đồ
    E4  max_features m: accuracy, sức mạnh, tương quan  EDA 5.3: "cây mạnh, tương quan thấp"
    E5  min_samples_leaf, max_depth; pixel hiếm         EDA 1.1: "xử lý bắt buộc"
    E6  Cấu hình cuối: train/val/test, lỗi theo cặp, độ quan trọng

Kết quả ghi vào results/random_forest/ theo quy ước chung (results/README.md):
    log.txt, summary.json      mọi bảng và số liệu thí nghiệm, gồm mọi số liệu mà hình cần
    final_metrics.json         mô hình cuối, cấu trúc chung để compare_models đọc
    figures/                   hp_e1..e5, interp_importance, eval_confusion, eval_confident_errors: do
                               src/plots/random_forest.py vẽ từ summary.json và final_metrics.json (gọi ở cuối
                               script; vẽ lại riêng: python -m src.plots.random_forest)
Baseline results/nearest_centroid/ do src/experiments/baseline.py ghi. Hàm phân tích dùng chung ở
src/evaluation/analysis.py, giống hệt các mô hình khác.

Mô hình: RandomForestClassifier của scikit-learn (src/models/random_forest.py, nhánh ensemble dùng thư viện),
một cây đơn ở E0 là DecisionTreeClassifier. Thuật toán và các hàm phân tích rừng mô tả ở src/models/random_forest.py.

Tham khảo:
    [1] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
    [2] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.
        Springer, 2009. Chương 15.
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.config import EDA_CLOSE_PAIRS, PERM_REPEATS, RARE_FREQ, SEED
from src.data.features import PixelMask, TreeFeatures
from src.data.preprocess import load_splits
from src.evaluation.analysis import (block_ids, block_permutation_importance, confident_error_records, log_loss,
                                     overconfident_error_rate, pair_errors, probabilistic_scores, share_in_pairs,
                                     top_confused_pairs)
from src.evaluation.metrics import classification_scores, evaluate
from src.evaluation.results_io import Logger, format_confusion, model_dir, rel_path, save_final_metrics, write_json
from src.models.nearest_centroid import NearestCentroidNP
from src.models.pipeline import Pipeline
from src.models.random_forest import (make_random_forest, staged_accuracy, strength_correlation,
                                      tree_stats)
from src.plots.random_forest import plot_all

KEY = "random_forest"
OUT_DIR = model_dir(KEY)

# Cấu hình xuất phát, theo đúng mục 5.3 của EDA; các thí nghiệm lần lượt cập nhật nó.
# sklearn luôn gộp cây bằng trung bình xác suất (soft voting); hard voting chỉ tính để so ở E3.
BASE_CONFIG = dict(mask="none", n_bins=16, n_estimators=25, max_features="sqrt", criterion="gini",
                   min_samples_leaf=1, max_depth=None)


# ---------------------------------------------------------------- tiện ích
def make_forest(cfg, seed=SEED, n_jobs=-1):
    """Pipeline: tiền xử lý cho cây (bỏ pixel, lượng tử hoá) rồi Random Forest, theo một cấu hình."""
    features = TreeFeatures(mask=cfg["mask"], n_bins=cfg["n_bins"])
    model = make_random_forest(n_estimators=cfg["n_estimators"], criterion=cfg["criterion"],
                               max_features=cfg["max_features"], max_depth=cfg["max_depth"],
                               min_samples_leaf=cfg["min_samples_leaf"], n_jobs=n_jobs, seed=seed)
    return Pipeline(features, model, name="Random Forest")


def grow_with_oob_curve(cfg, Z, y, seed, n_jobs, marks):
    """Trồng rừng tăng dần qua các mốc số cây và ghi OOB accuracy ở mỗi mốc.

    Z là ma trận đã qua TreeFeatures. Dùng warm_start=True: mỗi lần tăng n_estimators, sklearn chỉ trồng
    thêm các cây mới (cây cũ giữ nguyên) rồi tính lại oob_score_ trên cả rừng. Trả về rừng ở mốc cuối và
    danh sách OOB accuracy theo marks.
    """
    rf = make_forest(dict(cfg, n_estimators=marks[0]), seed, n_jobs).model
    rf.set_params(warm_start=True)
    oob = []
    for t in marks:
        rf.set_params(n_estimators=t)
        rf.fit(Z, y)
        oob.append(float(rf.oob_score_))
    return rf, oob


def run_forest(cfg, splits, seed=SEED, n_jobs=-1, keep=False):
    """Huấn luyện một rừng trên train, trả về các độ đo trên train/val (không đụng test)."""
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    pipe = make_forest(cfg, seed, n_jobs)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    rf = pipe.model
    P_va = pipe.predict_proba(X_va)
    va = classification_scores(y_va, np.argmax(P_va, axis=1))
    depth, leaves = tree_stats(rf)
    row = dict(cfg, seed=seed,
               val_acc=va["accuracy"], val_f1=va["f1_macro"], val_logloss=log_loss(y_va, P_va),
               oob=float(rf.oob_score_), fit_s=fit_s, n_features=int(rf.n_features_in_),
               m=int(rf.estimators_[0].max_features_), depth=depth, leaves=leaves)
    if keep:
        row["_pipe"] = pipe
    return row


def summarize(rows, key, metrics=("val_acc", "oob", "fit_s")):
    """Bảng "trung bình ± độ lệch chuẩn" theo giá trị của key (như bảng ở các bài thực hành)."""
    groups = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    head = f"{key:>12} | " + " | ".join(f"{m:>17}" for m in metrics) + " | số đặc trưng |  m"
    lines = [head, "-" * len(head)]
    stats = {}
    for g, rs in groups.items():
        cells, stats[g] = [], {}
        for m in metrics:
            v = np.array([r[m] for r in rs], dtype=np.float64)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            stats[g][m] = (float(v.mean()), float(sd))
            fmt = ".2f" if m.endswith("_s") else ".4f"
            cells.append(f"{v.mean():{fmt}} ± {sd:{fmt}}".rjust(17))
        lines.append(f"{g:>12} | " + " | ".join(cells) + f" | {rs[0]['n_features']:>12} | {rs[0]['m']:>2}")
    return "\n".join(lines), stats


def pooled_seed_std(rows, key, metric="val_acc"):
    """Độ lệch chuẩn do seed, gộp qua mọi cấu hình (ước lượng nhiễu khi so hai cấu hình)."""
    groups = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r[metric])
    ss, df = 0.0, 0
    for v in groups.values():
        v = np.asarray(v)
        if len(v) > 1:
            ss += ((v - v.mean()) ** 2).sum()
            df += len(v) - 1
    return float(np.sqrt(ss / df)) if df > 0 else 0.0


def pick_cheapest(stats, cost, tol):
    """Trong các lựa chọn có val_acc ≥ tốt nhất − tol, chọn lựa chọn có chi phí nhỏ nhất."""
    best = max(s["val_acc"][0] for s in stats.values())
    ok = [g for g, s in stats.items() if s["val_acc"][0] >= best - tol]
    return min(ok, key=cost)


# ---------------------------------------------------------------- các thí nghiệm
def run_e0(splits, log):
    """Mốc so sánh: tâm gần nhất và một cây CART đầy đủ (để thấy rừng giảm phương sai)."""
    log("\n=== E0. Mốc so sánh ===")
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    nc = NearestCentroidNP().fit(X_tr, y_tr)
    nc_acc = float((nc.predict(X_va) == y_va).mean())
    feats = TreeFeatures("none", 16).fit(X_tr)
    t0 = time.perf_counter()
    tree = DecisionTreeClassifier(criterion="gini", random_state=SEED).fit(feats.transform(X_tr), y_tr)
    tree_s = time.perf_counter() - t0
    tr_acc = float((tree.predict(feats.transform(X_tr)) == y_tr).mean())
    tree_acc = float((tree.predict(feats.transform(X_va)) == y_va).mean())
    leaves, depth = int(tree.get_n_leaves()), int(tree.get_depth())
    log(f"Tâm gần nhất          : val acc = {nc_acc:.4f}")
    log(f"Một cây CART đầy đủ   : train acc = {tr_acc:.4f} | val acc = {tree_acc:.4f} | "
        f"{leaves} lá, sâu {depth} | {tree_s:.1f} s")
    return dict(nearest_centroid_val=nc_acc, single_tree=dict(train_acc=tr_acc, val_acc=tree_acc,
                leaves=leaves, depth=depth, fit_s=tree_s))


def run_e1(splits, cfg, seeds, n_jobs, log, bins_grid):
    """Số ngưỡng ứng viên: EDA nói dữ liệu gần nhị phân nên chỉ cần ít ngưỡng."""
    log("\n=== E1. Số ngưỡng ứng viên mỗi pixel (n_bins) ===")
    rows = [run_forest(dict(cfg, n_bins=b), splits, s, n_jobs) for b in bins_grid for s in seeds]
    table, stats = summarize(rows, "n_bins")
    log(table)
    noise = pooled_seed_std(rows, "n_bins")
    log(f"Độ lệch chuẩn do seed (gộp): {noise:.4f}")
    chosen = int(pick_cheapest(stats, cost=lambda g: int(g), tol=2 * noise))
    log(f"Chọn n_bins = {chosen} (ít bin nhất trong các lựa chọn cách tốt nhất không quá 2 × độ lệch chuẩn)")
    return dict(rows=rows, stats=stats, noise=noise, chosen=chosen, n_estimators=cfg["n_estimators"],
                n_seeds=len(seeds))


def run_e2(splits, cfg, seeds, n_jobs, log, masks):
    """Bỏ pixel. Với m = √d thì bỏ pixel cũng đổi m, nên thêm một dòng đối chứng giữ m = 24."""
    log("\n=== E2. Bỏ pixel ===")
    rows, keep = [], {}
    for mk in masks:
        for s in seeds:
            r = run_forest(dict(cfg, mask=mk), splits, s, n_jobs, keep=(mk == "none" and s == seeds[0]))
            if "_pipe" in r:
                keep["none"] = r.pop("_pipe")
            rows.append(r)
    for s in seeds:                                                   # đối chứng: 784 pixel nhưng m = 24
        r = run_forest(dict(cfg, mask="none", max_features=24), splits, s, n_jobs)
        r["mask"] = "none, m=24"
        rows.append(r)
    table, stats = summarize(rows, "mask")
    log(table)
    noise = pooled_seed_std(rows, "mask")
    log(f"Độ lệch chuẩn do seed (gộp): {noise:.4f}")
    base_t = stats["none"]["fit_s"][0]
    for g, s in stats.items():
        log(f"  {g:>11}: thời gian = {s['fit_s'][0] / base_t:.2f} × none")
    nfeat = {r["mask"]: r["n_features"] for r in rows}
    cand = {g: s for g, s in stats.items() if g in masks}
    chosen = pick_cheapest(cand, cost=lambda g: nfeat[g], tol=2 * noise)
    log(f"Chọn mask = {chosen} (ít đặc trưng nhất trong các lựa chọn cách tốt nhất không quá 2 × độ lệch chuẩn)")
    X_tr = splits["train"][0]
    kinds = np.zeros(X_tr.shape[1], dtype=int)                       # 0: có mực, 1: gần như chết, 2: dead
    kinds[~PixelMask("rare").fit(X_tr).keep_] = 1
    kinds[~PixelMask("dead").fit(X_tr).keep_] = 2
    return dict(rows=rows, stats=stats, noise=noise, chosen=chosen, pixel_kinds=kinds), keep.get("none")


def run_e3(splits, cfg, seeds, n_jobs, log, t_max):
    """Accuracy theo số cây, nhiều seed. Validation: mọi T (cộng dồn phiếu từng cây), soft và hard voting.
    OOB: chỉ ở các mốc T trong `marks` (mỗi mốc sklearn tính lại OOB trên cả rừng)."""
    log("\n=== E3. Số cây T ===")
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    marks = sorted({t for t in (1, 5, 10, 25, 50, 100, 150, 200, 300, 500) if t <= t_max} | {t_max})
    feats = TreeFeatures(mask=cfg["mask"], n_bins=cfg["n_bins"]).fit(X_tr)
    Z_tr, Z_va = feats.transform(X_tr), feats.transform(X_va)
    curves = {"oob_soft": [], "val_soft": [], "val_hard": []}
    fit_s = []
    for s in seeds:
        t0 = time.perf_counter()
        rf, oob = grow_with_oob_curve(cfg, Z_tr, y_tr, s, n_jobs, marks)
        fit_s.append(time.perf_counter() - t0)
        curves["oob_soft"].append(oob)
        curves["val_soft"].append(staged_accuracy(rf, Z_va, y_va, "soft"))
        curves["val_hard"].append(staged_accuracy(rf, Z_va, y_va, "hard"))
    C = {k: np.asarray(v) for k, v in curves.items()}
    T = np.arange(1, t_max + 1)
    col = {"oob_soft": {t: i for i, t in enumerate(marks)}, "val_soft": {t: t - 1 for t in T},
           "val_hard": {t: t - 1 for t in T}}                       # cột ứng với T trong từng đường
    log(f"{'T':>5} | {'OOB (soft)':>17} | {'val (soft)':>17} | {'val (hard)':>17}")
    for t in marks:
        cells = []
        for k in ("oob_soft", "val_soft", "val_hard"):
            v = C[k][:, col[k][t]]
            cells.append(f"{v.mean():.4f} ± {v.std(ddof=1) if len(seeds) > 1 else 0:.4f}")
        log(f"{t:>5} | " + " | ".join(c.rjust(17) for c in cells))
    final = C["val_soft"][:, -1].mean()
    t_sat = int(T[np.argmax(C["val_soft"].mean(0) >= final - 0.001)])
    log(f"Val (soft) đạt trong 0,1 điểm của giá trị tại T = {t_max} từ T = {t_sat}")
    log(f"Thời gian trồng {t_max} cây, kể cả tính OOB ở {len(marks)} mốc: {np.mean(fit_s):.1f} s (n_jobs = {n_jobs})")
    gap = (C["val_soft"] - C["val_hard"]).mean(0)
    log(f"Soft − hard trên val: T = 10: {gap[min(9, t_max - 1)]:+.4f} | T = {t_max}: {gap[-1]:+.4f}")
    t_a = 25 if 25 in marks else marks[len(marks) // 2]
    oob_minus_val = {t: float((C["oob_soft"][:, col["oob_soft"][t]] - C["val_soft"][:, t - 1]).mean())
                     for t in marks}
    log(f"OOB − val: T = {t_a}: {oob_minus_val[t_a]:+.4f} | T = {t_max}: {oob_minus_val[t_max]:+.4f} "
        f"(mỗi mẫu OOB chỉ được ≈ 0,37·T cây bỏ phiếu, nên OOB ở T nhỏ đánh giá thấp rừng T cây)")
    return dict(curves=C, marks=marks, t_sat=t_sat, fit_s=float(np.mean(fit_s)), t_max=t_max,
                val_at={t: float(C["val_soft"][:, t - 1].mean()) for t in marks},
                val_sd_at={t: float(C["val_soft"][:, t - 1].std(ddof=1)) if len(seeds) > 1 else 0.0 for t in marks},
                oob_at={t: float(C["oob_soft"][:, col["oob_soft"][t]].mean()) for t in marks},
                oob_minus_val=oob_minus_val)


def run_e4(splits, cfg, n_jobs, log, m_grid, n_trees):
    """max_features: m nhỏ làm cây yếu nhưng ít tương quan; m = d là bagging thuần [1]."""
    log(f"\n=== E4. max_features m ({n_trees} cây mỗi cấu hình) ===")
    X_va, y_va = splits["val"]
    # sklearn báo lỗi nếu max_features > số pixel giữ lại d, nên cắt lưới về d (m = d: bagging thuần)
    d = PixelMask(cfg["mask"]).fit(splits["train"][0]).keep_.sum()
    rows = []
    for m in sorted({min(int(m), int(d)) for m in m_grid}):
        r = run_forest(dict(cfg, max_features=m, n_estimators=n_trees), splits, SEED, n_jobs, keep=True)
        pipe = r.pop("_pipe")
        r["strength"], r["correlation"] = strength_correlation(pipe.model, pipe.features.transform(X_va), y_va)
        rows.append(r)
        log(f"m = {r['m']:>4}: val acc = {r['val_acc']:.4f} | acc 1 cây (TB) = {r['strength']:.4f} | "
            f"tương quan lỗi giữa 2 cây = {r['correlation']:.3f} | sâu {r['depth']:.1f} | {r['fit_s']:.1f} s")
    return dict(rows=rows, n_trees=n_trees)


def run_e5(splits, cfg, n_jobs, log, leaf_grid, depth_grid, forest_none):
    """Giới hạn lá / độ sâu. EDA cho rằng bắt buộc để không học từ 116 pixel gần như chết."""
    log("\n=== E5. min_samples_leaf và max_depth ===")
    X_tr, y_tr = splits["train"]
    rows = []
    for msl in leaf_grid:
        r = run_forest(dict(cfg, min_samples_leaf=msl), splits, SEED, n_jobs, keep=True)
        pipe = r.pop("_pipe")
        r["train_acc"] = float((pipe.predict(X_tr) == y_tr).mean())
        r["vary"] = f"min_samples_leaf={msl}"
        rows.append(r)
    for dmax in depth_grid:
        r = run_forest(dict(cfg, max_depth=dmax), splits, SEED, n_jobs, keep=True)
        pipe = r.pop("_pipe")
        r["train_acc"] = float((pipe.predict(X_tr) == y_tr).mean())
        r["vary"] = f"max_depth={dmax}"
        rows.append(r)
    for r in rows:
        log(f"{r['vary']:>20}: train acc = {r['train_acc']:.4f} | val acc = {r['val_acc']:.4f} | "
            f"val log-loss = {r['val_logloss']:.4f} | {r['leaves']:.0f} lá/cây, sâu {r['depth']:.1f} | {r['fit_s']:.1f} s")

    # Pixel gần như chết được dùng để chia nhánh bao nhiêu (rừng mask="none" của E2)
    rare = {}
    if forest_none is not None:
        freq = forest_none.features.mask_.freq_
        is_rare = (freq > 0) & (freq < RARE_FREQ)
        n_split = n_rare = 0
        small_child, sizes, imp_rare = [], [], []
        for est in forest_none.model.estimators_:
            # est.tree_: mảng theo nút. feature < 0 là lá; weighted_n_node_samples đếm số ảnh ở nút
            # theo mẫu bootstrap (ảnh rút 2 lần tính 2), children_left/right là chỉ số hai nút con.
            t = est.tree_
            inner = t.feature >= 0
            f = t.feature[inner]
            n_split += inner.sum()
            hit = is_rare[f]
            n_rare += hit.sum()
            nodes = np.flatnonzero(inner)[hit]
            n_node = t.weighted_n_node_samples
            sizes += list(n_node[nodes])
            small_child += list(np.minimum(n_node[t.children_left[nodes]], n_node[t.children_right[nodes]]))
            imp_rare.append(est.feature_importances_[is_rare].sum())
        rare = dict(share_splits=float(n_rare / n_split), share_mdi=float(np.mean(imp_rare)),
                    median_node=float(np.median(sizes)) if sizes else None,
                    median_small_child=float(np.median(small_child)) if small_child else None,
                    share_pixels=float(is_rare.mean()))
        log(f"Pixel gần như chết ({is_rare.sum()} vị trí, {is_rare.mean():.1%} số pixel): "
            f"{rare['share_splits']:.2%} số phép chia, {rare['share_mdi']:.2%} tổng MDI; "
            f"trung vị số mẫu ở nút = {rare['median_node']:.1f}, ở nút con nhỏ hơn = {rare['median_small_child']:.1f}")
    return dict(rows=rows, rare_pixels=rare, n_estimators=cfg["n_estimators"])


def run_e6(splits, cfg, n_jobs, log, perm_repeats, quick=False):
    """Cấu hình cuối: huấn luyện trên train, đánh giá train/val/test (test dùng một lần)."""
    log("\n=== E6. Cấu hình cuối ===")
    log("Cấu hình: " + json.dumps(cfg, ensure_ascii=False))
    pipe = make_forest(cfg, SEED, n_jobs)
    res = evaluate(pipe, splits)
    rf, feats = pipe.model, pipe.features
    # fit_time gồm cả bước tính OOB: sklearn tính OOB một lần ngay trong fit (oob_score=True)
    n_used = os.cpu_count() if n_jobs in (-1, None) else n_jobs
    log(f"Huấn luyện: {res['fit_time']:.1f} s ({n_used} luồng, kể cả tính OOB) | OOB acc = {rf.oob_score_:.4f}")
    for name in ("train", "val", "test"):
        s = res[name]
        log(f"{name:>5}: acc {s['accuracy']:.4f} | P {s['precision_macro']:.4f} | R {s['recall_macro']:.4f} | "
            f"F1 {s['f1_macro']:.4f} | dự đoán {s['predict_time']:.2f} s")

    depth, leaves = tree_stats(rf)
    out = dict(config=cfg, fit_s=res["fit_time"], oob=float(rf.oob_score_), depth=depth, leaves=leaves)
    for name in ("train", "val", "test"):
        log(format_confusion(res[name]["confusion_matrix"], f"\n[{name}] ma trận nhầm lẫn (hàng: nhãn thật, cột: dự đoán)"))

    nc = NearestCentroidNP().fit(*splits["train"])
    prob = {}
    for name in ("train", "val", "test"):
        X, y = splits[name]
        s = res[name]
        P = pipe.predict_proba(X)
        prob[name] = probabilistic_scores(y, P)
        cm = s["confusion_matrix"]
        cm_nc = classification_scores(y, nc.predict(X))["confusion_matrix"]
        out[name] = dict(accuracy=s["accuracy"], precision_macro=s["precision_macro"],
                         recall_macro=s["recall_macro"], f1_macro=s["f1_macro"], predict_s=s["predict_time"],
                         per_class=s["per_class"], confusion_matrix=cm,
                         log_loss=log_loss(y, P), overconfident_099=overconfident_error_rate(y, P, 0.99),
                         mean_conf_errors=float(P.max(1)[np.argmax(P, 1) != y].mean())
                         if (np.argmax(P, 1) != y).any() else float("nan"),
                         pairs=pair_errors(cm, EDA_CLOSE_PAIRS), share_close_pairs=share_in_pairs(cm),
                         top_pairs=top_confused_pairs(cm, 5),
                         nearest_centroid=dict(accuracy=float(np.trace(cm_nc) / cm_nc.sum()),
                                               pairs=pair_errors(cm_nc, EDA_CLOSE_PAIRS),
                                               share_close_pairs=share_in_pairs(cm_nc)))
    for name in ("val", "test"):
        o = out[name]
        log(f"[{name}] log-loss {o['log_loss']:.4f} | sai mà p > 0,99: {o['overconfident_099']:.2%} | "
            f"p trung bình của mẫu sai: {o['mean_conf_errors']:.3f}")
        log(f"[{name}] Lỗi ở 5 cặp gần nhất chiếm {o['share_close_pairs']:.1%} số lỗi (rải đều: {5 / 45:.1%}); "
            f"lỗi theo cặp RF {o['pairs']}")
        log(f"[{name}]     5 cặp nhầm nhiều nhất: {o['top_pairs']}")
        log(f"[{name}] So với tâm gần nhất: acc {o['nearest_centroid']['accuracy']:.4f}, lỗi theo cặp "
            f"{o['nearest_centroid']['pairs']}")
        red = {k: 1 - o["pairs"][k] / max(o["nearest_centroid"]["pairs"][k], 1) for k in o["pairs"]}
        n_err_rf = o["confusion_matrix"].sum() - np.trace(o["confusion_matrix"])
        n_err_nc = (1 - o["nearest_centroid"]["accuracy"]) * o["confusion_matrix"].sum()
        log(f"[{name}]     RF giảm lỗi so với tâm gần nhất: toàn bộ {1 - n_err_rf / n_err_nc:.1%}; theo cặp "
            + ", ".join(f"{k}: {v:.1%}" for k, v in red.items()))

    # Độ quan trọng: bản đồ MDI (hình riêng của RF) và hoán vị khối 4×4 (chung cho mọi mô hình).
    # So với Fisher (H4) làm một lần cho mọi mô hình ở compare_models.
    X_tr, _ = splits["train"]
    freq = (X_tr > 0).mean(axis=0)
    mdi = feats.to_image(rf.feature_importances_, fill=0.0).ravel()
    X_va, y_va = splits["val"]
    t0 = time.perf_counter()
    drops = block_permutation_importance(pipe, X_va, y_va, n_repeats=perm_repeats, seed=SEED)
    ids = block_ids()
    log(f"Hoán vị khối 4×4 trên val ({time.perf_counter() - t0:.0f} s): khối quan trọng nhất làm giảm acc "
        f"{drops.max():.4f}")

    # final_metrics.json: cấu trúc chung, để compare_models đọc.
    save_final_metrics(
        "random_forest", "Random Forest", res, cfg, probabilistic=prob,
        block_importance=drops, n_jobs=n_jobs,
        extra=dict(oob_accuracy=float(rf.oob_score_), mean_depth=out["depth"], mean_leaves=out["leaves"]),
        quick=quick)

    # Số liệu cho hình giải thích (interp_importance.png) và 20 lỗi tự tin nhất trên test
    out["importance_maps"] = dict(ink_freq=freq, mdi=np.where(mdi == 0, np.nan, mdi),   # NaN: không bao giờ chia
                                  block=drops[ids])
    out["confident_errors"] = confident_error_records(splits["test"][1], pipe.predict_proba(splits["test"][0]))
    return out


# ---------------------------------------------------------------- chương trình chính
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="lưới nhỏ, ít cây, một seed: chỉ để kiểm tra code")
    ap.add_argument("--n-jobs", type=int, default=-1, help="số luồng trồng cây song song (-1 = mọi lõi)")
    args = ap.parse_args()

    log = Logger(OUT_DIR / "log.txt")
    q = args.quick
    seeds = (0,) if q else (0, 1)
    seeds_curve = (0,) if q else (0, 1, 2)
    t0 = time.perf_counter()
    splits = load_splits()
    log(f"train {splits['train'][0].shape} | val {splits['val'][0].shape} | test {splits['test'][0].shape}"
        f" | n_jobs = {args.n_jobs} | quick = {q}")

    cfg = dict(BASE_CONFIG, n_estimators=8 if q else BASE_CONFIG["n_estimators"])
    summary = {"base_config": dict(cfg)}
    summary["E0"] = run_e0(splits, log)

    e1 = run_e1(splits, cfg, seeds, args.n_jobs, log, (2, 4, 16) if q else (2, 4, 8, 16, 256))
    cfg["n_bins"] = e1["chosen"]
    summary["E1"] = e1

    e2, forest_none = run_e2(splits, cfg, seeds, args.n_jobs, log,
                               ("none", "rare", "crop24") if q else ("none", "dead", "rare", "crop24", "crop20"))
    cfg["mask"] = e2["chosen"]
    summary["E2"] = e2

    e3 = run_e3(splits, cfg, seeds_curve, args.n_jobs, log, t_max=20 if q else 200)
    summary["E3"] = e3

    e4 = run_e4(splits, cfg, args.n_jobs, log, (4, 24, 96) if q else (1, 4, 8, 16, 24, 48, 96, 192, 576),
                         n_trees=5 if q else 10)
    summary["E4"] = e4
    # Sàng lọc ở T nhỏ, rồi xác nhận ứng viên tốt nhất với √d ở T lớn hơn, nhiều seed
    best_m = max(e4["rows"], key=lambda r: r["val_acc"])
    m_sqrt = int(np.sqrt(best_m["n_features"]))
    if best_m["m"] != m_sqrt:
        t_conf = 10 if q else 50
        log(f"Xác nhận m = {best_m['m']} so với m = √d = {m_sqrt} ở T = {t_conf}, {len(seeds)} seed:")
        conf = [run_forest(dict(cfg, max_features=m, n_estimators=t_conf), splits, s, args.n_jobs)
                for m in (m_sqrt, int(best_m["m"])) for s in seeds]
        table, st = summarize(conf, "max_features", metrics=("val_acc", "oob", "fit_s"))
        log(table)
        noise = max(pooled_seed_std(conf, "max_features"), e1["noise"])
        gain = st[str(int(best_m["m"]))]["val_acc"][0] - st[str(m_sqrt)]["val_acc"][0]
        cost = st[str(int(best_m["m"]))]["fit_s"][0] / st[str(m_sqrt)]["fit_s"][0]
        log(f"m = {best_m['m']} hơn √d {gain:+.4f} (ngưỡng 2 × độ lệch chuẩn = {2 * noise:.4f}), thời gian × {cost:.2f}")
        summary["E4"]["confirm"] = dict(rows=conf, stats=st, gain=gain, time_ratio=cost, noise=noise)
        if gain > 2 * noise:
            cfg["max_features"] = int(best_m["m"])
    log(f"Chọn max_features = {cfg['max_features']}")

    e5 = run_e5(splits, cfg, args.n_jobs, log, (1, 3) if q else (1, 3, 10, 30),
                       (12,) if q else (10, 15, 20), forest_none)
    summary["E5"] = e5
    noise5 = e1["noise"]
    best5 = max(e5["rows"], key=lambda r: r["val_acc"])
    base5 = e5["rows"][0]                                             # min_samples_leaf = 1, không giới hạn độ sâu
    if best5["val_acc"] - base5["val_acc"] > 2 * noise5:
        cfg["min_samples_leaf"], cfg["max_depth"] = best5["min_samples_leaf"], best5["max_depth"]
    log(f"Chọn min_samples_leaf = {cfg['min_samples_leaf']}, max_depth = {cfg['max_depth']}")

    # Số cây: bội số 50 nhỏ nhất không dưới điểm bão hoà của E3 (thêm cây không hại, chỉ tốn thời gian)
    cfg["n_estimators"] = 20 if q else int(max(50, 50 * np.ceil(e3["t_sat"] / 50)))
    log(f"Chọn n_estimators = {cfg['n_estimators']} (E3: bão hoà từ T = {e3['t_sat']})")
    summary["final"] = run_e6(splits, cfg, args.n_jobs, log, perm_repeats=1 if q else PERM_REPEATS, quick=q)
    summary["total_s"] = time.perf_counter() - t0
    log(f"\nTổng thời gian: {summary['total_s'] / 60:.1f} phút")
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'summary.json')}, log.txt và figures/")


if __name__ == "__main__":
    main()