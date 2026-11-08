"""So sánh các mô hình đã chạy xong, từ results/<mô hình>/final_metrics.json (quy ước: results/README.md).

Chạy từ gốc repo, sau khi đã chạy baseline, script của từng mô hình (và tuỳ chọn sklearn_reference, ensemble):
    python -m src.experiments.compare_models

Mô hình nào chưa có final_metrics.json thì bỏ qua, nên chạy lại được mỗi khi có thêm mô hình.
Kết quả ghi vào results/comparison/:
    log.txt                    bảng so sánh, trạng thái giả thuyết H1–H4 của EDA 5.6
    comparison.csv             một hàng cho mỗi (mô hình, tập)
    summary.json               số liệu cho hình so sánh (độ đo, thời gian, lỗi theo cặp, độ quan trọng theo khối)
    figures/cmp_*.png          do src/plots/compare_models.py vẽ từ summary.json (gọi ở cuối script;
                               vẽ lại riêng: python -m src.plots.compare_models)

Theo EDA 5.6, giả thuyết được kiểm chứng trên VALIDATION; số trên test chỉ để báo cáo.
CNN và ensemble (results/ensemble/final_metrics.json) là hướng nâng cao, xếp sau các mô hình của EDA 5.6.
Mọi phép tính dùng hàm trong src/evaluation/analysis.py, giống hệt các script mô hình.
"""

import csv
import json

import numpy as np

from src.config import BASELINE_KEY, EDA_CLOSE_PAIRS, ENSEMBLE_KEY, MODEL_KEYS, SPLITS
from src.evaluation.analysis import (N_PAIRS, block_sums, fisher_scores, pair_errors, spearman,
                                     top_confused_pairs)
from src.evaluation.results_io import RESULTS_DIR, Logger, load_final_metrics, rel_path, write_json
from src.plots.compare_models import plot_all

KEY = "comparison"
OUT_DIR = RESULTS_DIR / KEY
ALL_KEYS = (BASELINE_KEY, *MODEL_KEYS, ENSEMBLE_KEY)
ADVANCED = ("cnn", ENSEMBLE_KEY)  # hướng nâng cao, ngoài giả thuyết EDA 5.6
LINEAR, NONLINEAR = ("naive_bayes", "logistic_regression"), ("random_forest", "mlp")
RHO_MIN = 0.7  # H4: tương quan hạng "mạnh" giữa độ quan trọng theo khối và Fisher theo khối


def fmt(v, spec=".4f"):
    return "—" if v is None else format(v, spec)


def fisher_map():
    """Fisher score theo pixel trên splits["train"], cùng hàm và ngưỡng 0,1% với EDA 3.3."""
    from src.data.preprocess import load_splits

    X, y = load_splits()["train"]
    return fisher_scores(X, y)


def as_array(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


# ---------------------------------------------------------------------------
def main():
    log = Logger(OUT_DIR / "log.txt")
    runs = {}
    for key in ALL_KEYS:
        r = load_final_metrics(key)
        if r is not None:
            runs[key] = r
    missing = [k for k in ALL_KEYS if k not in runs]
    log(f"Đã có: {', '.join(runs[k]['display_name'] for k in runs)}")
    if missing:
        log(f"Chưa có (bỏ qua): {', '.join(missing)}"
            + ("  -> chạy python -m src.experiments.baseline" if BASELINE_KEY in missing else ""))
    quick = [runs[k]["display_name"] for k in runs if runs[k].get("quick")]
    if quick:
        log(f"CẢNH BÁO: {', '.join(quick)} là kết quả chạy --quick, không dùng cho báo cáo.")
    if len(runs) < 2:
        log("Cần ít nhất 2 mô hình để so sánh.")
        return
    keys = list(runs)
    names = {k: runs[k]["display_name"] for k in keys}

    # --- 1. Bảng độ đo ---
    rows = []
    for split in SPLITS:
        log(f"\n[{split}]")
        log(f"{'Mô hình':<26}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'log-loss':>10}"
            f"{'sai p>.99':>10}{'Fit(s)':>9}{'Pred(s)':>9}")
        for k in keys:
            s = runs[k]["splits"][split]
            p = s.get("probabilistic") or {}
            log(f"{names[k]:<26}{s['accuracy']:>8.4f}{s['precision_macro']:>8.4f}{s['recall_macro']:>8.4f}"
                f"{s['f1_macro']:>8.4f}{fmt(p.get('log_loss')):>10}"
                f"{fmt(p.get('overconfident_errors'), '.3%'):>10}{fmt(runs[k].get('fit_time'), '.1f'):>9}"
                f"{fmt(s.get('predict_time'), '.3f'):>9}")
            rows.append({"model": k, "split": split, "accuracy": s["accuracy"], "precision_macro": s["precision_macro"],
                         "recall_macro": s["recall_macro"], "f1_macro": s["f1_macro"],
                         "log_loss": p.get("log_loss"),
                         "overconfident_errors": p.get("overconfident_errors"),
                         "fit_time": runs[k].get("fit_time"), "predict_time": s.get("predict_time"),
                         "n_jobs": runs[k].get("runtime", {}).get("n_jobs")})
    with open(OUT_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    jobs = {k: runs[k].get("runtime", {}).get("n_jobs") for k in keys}
    if len({j for j in jobs.values() if j is not None}) > 1:
        log("\nLƯU Ý thời gian: các mô hình chạy với số luồng khác nhau ("
            + ", ".join(f"{names[k]}: {jobs[k]}" for k in keys) + "). So thời gian huấn luyện phải kèm điều kiện này.")

    # --- 2. Lỗi theo cặp ---
    log(f"\nLỗi ở 5 cặp EDA 3.1 (cộng hai chiều) trên val | tỉ lệ trong tổng lỗi (rải đều: {5 / N_PAIRS:.1%}):")
    log(f"{'Mô hình':<26}" + "".join(f"{f'{i}–{j}':>7}" for i, j in EDA_CLOSE_PAIRS) + f"{'tổng lỗi':>10}{'5 cặp':>8}")
    pair_val = {}
    for k in keys:
        cm = np.asarray(runs[k]["splits"]["val"]["confusion_matrix"])
        pe = pair_errors(cm)
        total = int(cm.sum() - np.trace(cm))
        inside = sum(pe[p] for p in EDA_CLOSE_PAIRS)
        pair_val[k] = (pe, total)
        top5 = top_confused_pairs(cm, 5)
        log(f"{names[k]:<26}" + "".join(f"{pe[p]:>7}" for p in EDA_CLOSE_PAIRS) + f"{total:>10}{inside / total:>8.1%}"
            f"   5 cặp nhầm nhiều nhất: {', '.join(f'{i}–{j}' for i, j, _ in top5)}")

    # --- 3. Trạng thái giả thuyết H1–H3 (trên val); H4 ở mục 4 ---
    log("\nTrạng thái giả thuyết EDA 5.6 (kiểm chứng trên val; tâm gần nhất chỉ là mốc):")
    v = {k: runs[k]["splits"]["val"] for k in keys}

    def has(*ks):
        return all(k in runs for k in ks)

    if has(*LINEAR):
        nb, lr = v["naive_bayes"], v["logistic_regression"]
        pn, pl = nb.get("probabilistic") or {}, lr.get("probabilistic") or {}
        acc_ok = lr["accuracy"] > nb["accuracy"]
        prob_ok = (pl.get("log_loss") is not None and pn.get("log_loss") is not None
                   and pl["log_loss"] < pn["log_loss"] and pl["overconfident_errors"] < pn["overconfident_errors"])
        log(f"  H1 (LR chính xác hơn và xác suất đáng tin hơn MNB): {'ủng hộ' if acc_ok and prob_ok else 'không ủng hộ'} "
            f"| acc {lr['accuracy']:.4f} vs {nb['accuracy']:.4f}; log-loss {fmt(pl.get('log_loss'))} vs "
            f"{fmt(pn.get('log_loss'))}; sai mà p>0,99 {fmt(pl.get('overconfident_errors'), '.3%')} vs "
            f"{fmt(pn.get('overconfident_errors'), '.3%')}")
    else:
        log(f"  H1: chờ kết quả {', '.join(k for k in LINEAR if k not in runs)}")
    if "logistic_regression" in runs:
        lr = v["logistic_regression"]
        for k in NONLINEAR:
            if k in runs:
                ok = v[k]["accuracy"] > lr["accuracy"] and v[k]["f1_macro"] > lr["f1_macro"]
                log(f"  H2 ({names[k]} vượt LR): {'ủng hộ' if ok else 'không ủng hộ'} | acc {v[k]['accuracy']:.4f} vs "
                    f"{lr['accuracy']:.4f}; F1 {v[k]['f1_macro']:.4f} vs {lr['f1_macro']:.4f}")
            else:
                log(f"  H2 ({k}): chờ kết quả")

    # H3 hai vế: (a) lỗi của mô hình tuyến tính DỒN vào 5 cặp: chiếm ≥ gấp đôi mức rải đều và ≥ 3/5 cặp nằm trong
    # top 5; (b) so với LR, mô hình phi tuyến giảm lỗi ở 5 cặp nhiều hơn mức giảm chung
    in_pairs = {k: sum(pair_val[k][0][p] for p in EDA_CLOSE_PAIRS) for k in keys}
    n_top = {k: len({(i, j) for i, j, _ in top_confused_pairs(runs[k]["splits"]["val"]["confusion_matrix"], 5)}
                    & set(EDA_CLOSE_PAIRS)) for k in keys}
    share = {k: in_pairs[k] / max(pair_val[k][1], 1) for k in keys}
    log(f"  H3 (a) lỗi của mô hình tuyến tính dồn vào 5 cặp EDA 3.1 (≥ {2 * 5 / N_PAIRS:.1%} lỗi, ≥ 3/5 cặp trong top 5):")
    part_a = []
    for k in (k for k in LINEAR if k in runs):
        part_a.append(share[k] >= 2 * 5 / N_PAIRS and n_top[k] >= 3)
        log(f"      {names[k]:<26} {share[k]:.1%} lỗi, {n_top[k]}/5 cặp trong top 5 -> {'đạt' if part_a[-1] else 'không đạt'}")
    part_b = []
    if "logistic_regression" in runs:
        log("  H3 (b) mô hình phi tuyến giảm lỗi ở 5 cặp nhiều hơn mức giảm chung (so với LR):")
        tot_lr = pair_val["logistic_regression"][1]
        for k in (k for k in NONLINEAR if k in runs):
            red_pairs = 1 - in_pairs[k] / max(in_pairs["logistic_regression"], 1)
            red_all = 1 - pair_val[k][1] / max(tot_lr, 1)
            part_b.append(red_pairs > red_all)
            log(f"      {names[k]:<26} giảm {red_pairs:.0%} ở 5 cặp, {red_all:.0%} tổng lỗi -> "
                f"{'đạt' if part_b[-1] else 'không đạt'}; lỗi còn lại: {share[k]:.1%} ở 5 cặp, {n_top[k]}/5 trong top 5")
    if part_a and part_b:
        verdict = "ủng hộ" if all(part_a) and all(part_b) else "ủng hộ một phần" if any(part_a + part_b) else "không ủng hộ"
        log(f"  H3: {verdict}")
    else:
        log("  H3: chờ kết quả")

    # Nâng cao (ngoài EDA 5.6): CNN và gộp mô hình, cùng số đo với H2, H3 nhưng không tính vào kết luận giả thuyết
    extra = [k for k in ADVANCED if k in runs]
    if extra and "logistic_regression" in runs:
        lr, tot_lr = v["logistic_regression"], pair_val["logistic_regression"][1]
        log("\nNâng cao (ngoài EDA 5.6, cùng số đo với H2, H3; không tính vào kết luận giả thuyết):")
        for k in extra:
            red_pairs = 1 - in_pairs[k] / max(in_pairs["logistic_regression"], 1)
            red_all = 1 - pair_val[k][1] / max(tot_lr, 1)
            log(f"  {names[k]:<26} acc {v[k]['accuracy']:.4f} (LR {lr['accuracy']:.4f}); so với LR giảm {red_pairs:.0%} "
                f"lỗi ở 5 cặp, {red_all:.0%} tổng lỗi; lỗi còn lại: {share[k]:.1%} ở 5 cặp, {n_top[k]}/5 trong top 5")
    if has("cnn", ENSEMBLE_KEY):
        ens, e_ens, e_cnn = ENSEMBLE_KEY, pair_val[ENSEMBLE_KEY][1], pair_val["cnn"][1]
        log(f"  {runs[ens]['config'].get('name')} so với CNN đơn: acc {v[ens]['accuracy']:.4f} vs "
            f"{v['cnn']['accuracy']:.4f}; lỗi trên val {e_ens} vs {e_cnn} ({e_ens / max(e_cnn, 1) - 1:+.0%}), "
            f"đổi lại thời gian huấn luyện {fmt(runs[ens].get('fit_time'), '.0f')} s vs "
            f"{fmt(runs['cnn'].get('fit_time'), '.0f')} s")

    # --- 4. H4, khả năng giải thích: số đo CHUNG theo khối, so với tổng Fisher của khối ---
    blk = {k: as_array(runs[k]["block_importance"]) for k in keys if runs[k].get("block_importance") is not None}
    block = None
    if blk:
        F_block = block_sums(fisher_map())
        log(f"\nH4 (số đo CHUNG cho mọi mô hình): accuracy giảm khi hoán vị khối 4×4 trên val. Spearman với "
            f"tổng Fisher của khối; ủng hộ khi ρ ≥ {RHO_MIN} ở mọi mô hình:")
        rb = {}
        for k, b in blk.items():
            rb[k] = spearman(b, F_block)
            log(f"  {names[k]:<26} ρ = {rb[k]:.3f} | khối quan trọng nhất làm giảm acc {np.nanmax(b):.4f}")
        rm = [r for k, r in rb.items() if k not in (BASELINE_KEY, *ADVANCED)]
        if rm:
            log(f"  H4: {'ủng hộ' if all(r >= RHO_MIN for r in rm) else 'không ủng hộ'} "
                f"(ρ khối của các mô hình từ {min(rm):.3f} đến {max(rm):.3f}; không tính mốc tâm gần nhất "
                f"và hướng nâng cao)")
        block = dict(fisher=F_block, models={k: dict(importance=blk[k], rho=rb[k]) for k in blk})

    # --- 5. Đối chiếu sklearn (rubric mục 2) ---
    ref_path = RESULTS_DIR / "sklearn_reference" / "summary.json"
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        log("\nĐối chiếu scikit-learn (cùng đặc trưng, cùng siêu tham số; src/experiments/sklearn_reference.py):")
        log(f"{'Mô hình':<26}{'test acc tự cài':>16}{'sklearn':>9}{'chênh':>8}{'fit tự cài':>12}{'sklearn':>9}")
        for k, r in ref.items():
            t = r["test"]
            log(f"{names.get(k, k):<26}{t['ours_accuracy']:>16.4f}{t['accuracy']:>9.4f}"
                f"{t['ours_accuracy'] - t['accuracy']:>+8.4f}{fmt(r['ours_fit_time'], '.1f'):>12}{r['fit_time']:>9.1f}")
    else:
        log("\n(Chưa có results/sklearn_reference/summary.json: chạy python -m src.experiments.sklearn_reference)")

    alt_path = RESULTS_DIR / "sklearn_reference" / "ensemble_alternatives.json"
    if alt_path.is_file():
        alt = json.loads(alt_path.read_text(encoding="utf-8"))
        log("\nPhương án ensemble bị loại (EDA 5.3; sklearn, cùng đặc trưng và số cây với RF, không tinh chỉnh):")
        log(f"{'Mô hình':<52}{'val acc':>9}{'test acc':>10}{'fit (s)':>9}{'luồng':>7}")
        for r in alt.values():
            log(f"{r['model']:<52}{r['val']['accuracy']:>9.4f}{r['test']['accuracy']:>10.4f}"
                f"{fmt(r['fit_time'], '.1f'):>9}{r['n_jobs']:>7}")

    # --- 6. Số liệu cho hình so sánh ---
    summary = dict(
        models=keys, names=names, n_jobs=jobs,
        metrics={k: dict(fit_time=runs[k].get("fit_time"),
                         **{sp: {m: runs[k]["splits"][sp].get(m) for m in ("accuracy", "f1_macro", "predict_time")}
                            for sp in SPLITS}) for k in keys},
        per_class_f1_test={k: runs[k]["splits"]["test"]["per_class"]["f1"] for k in keys},
        pair_errors_val={k: [pair_val[k][0][p] for p in EDA_CLOSE_PAIRS] for k in keys},
        block=block)
    write_json(summary, OUT_DIR / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(OUT_DIR / 'log.txt')}, {rel_path(OUT_DIR / 'comparison.csv')}, summary.json và figures/")


if __name__ == "__main__":
    main()
