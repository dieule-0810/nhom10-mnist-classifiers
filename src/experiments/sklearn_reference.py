"""Đối chiếu bản tự cài với scikit-learn trên MNIST, cùng đặc trưng và cùng siêu tham số (rubric mục 2).

tests/ đã đối chiếu từng phần trên dữ liệu nhỏ (gradient, nghiệm tối ưu, xác suất...).
Script này bổ sung bảng ở quy mô thật: lấy cấu hình cuối trong results/<key>/final_metrics.json,
dựng mô hình sklearn tương đương, huấn luyện trên splits["train"], so accuracy, F1 và thời gian.

Chạy từ gốc repo, SAU khi đã chạy script của các mô hình:
    python -m src.experiments.sklearn_reference                       # mọi mô hình đã có kết quả
    python -m src.experiments.sklearn_reference --models naive_bayes mlp

Ghi results/sklearn_reference/summary.json và log.txt; compare_models đọc file này. Chạy một phần mô hình thì
chỉ cập nhật các dòng đó trong summary.json, giữ nguyên các dòng khác.
Thêm mô hình mới: viết một hàm build_<key>(cfg) trả về (Pipeline, mô tả) và thêm vào BUILDERS.

Riêng LR có thêm một dòng "giải tới hội tụ": LogisticRegression (lbfgs) với cùng hàm mục tiêu, để biết dừng sớm
theo val loss có làm mất accuracy so với cực tiểu của J hay không (ghi vào khoá "converged" của dòng LR).

Random Forest không có trong bảng đối chiếu: mô hình của nhóm là RandomForestClassifier của sklearn (nhánh ensemble
dùng thư viện, xem src/models/random_forest.py).

Phương án ensemble bị loại (EDA 5.3): AdaBoost và Gradient Boosting của sklearn, cùng đặc trưng và cùng số cây với
Random Forest cuối, siêu tham số cố định (không tinh chỉnh). Ghi riêng vào ensemble_alternatives.json cạnh dòng
Random Forest, để biết chọn RF thì đánh đổi bao nhiêu accuracy và thời gian. Tắt bằng --no-alternatives.
"""

import argparse
import json
import os

from src.config import DISPLAY_NAMES, MODEL_KEYS, SEED
from src.data.features import TreeFeatures, make_features
from src.data.preprocess import load_splits
from src.evaluation.metrics import evaluate
from src.evaluation.results_io import Logger, load_final_metrics, model_dir, rel_path, write_json
from src.models.pipeline import Pipeline

OUT_KEY = "sklearn_reference"


def build_logistic_regression(cfg):
    """Softmax regression = MLPClassifier không lớp ẩn (đầu ra softmax), cùng điều kiện dừng với bản tự cài:
    mini-batch SGD + momentum (không Nesterov), cùng η, α, batch; huấn luyện đúng cfg["epochs"] epoch (epoch mà
    bản tự cài giữ lại khi dừng sớm), không dừng sớm vì sklearn chỉ dừng sớm trên tập con của train.

    sklearn.linear_model.LogisticRegression không có mini-batch, momentum hay dừng sớm (lbfgs giải tới hội tụ), nên
    dùng riêng ở build_logistic_regression_converged và trong tests/check_logistic_regression.py.
    L2: sklearn cộng alpha·W / batch_size vào gradient của loss trung bình, bản tự cài cộng λW, nên alpha = λ·batch;
    cả hai không phạt bias. Khác nhau còn lại: khởi tạo (sklearn Glorot đều, bản tự cài W = 0; J lồi) và dãy trộn.
    """
    from sklearn.neural_network import MLPClassifier

    epochs = cfg.get("epochs", cfg["max_epochs"])
    model = MLPClassifier(hidden_layer_sizes=(), solver="sgd", learning_rate="constant", learning_rate_init=cfg["eta"],
                          momentum=cfg["momentum"], nesterovs_momentum=False, batch_size=cfg["batch_size"],
                          alpha=cfg["lam"] * cfg["batch_size"], max_iter=epochs, n_iter_no_change=epochs + 1,
                          tol=0.0, shuffle=True, random_state=SEED)
    feats = make_features(cfg["mask"], cfg.get("scale", "scalar"))
    return Pipeline(feats, model), f"MLPClassifier(không lớp ẩn = softmax regression; sgd, {epochs} epoch)"


def build_naive_bayes(cfg):
    """Cùng mặt nạ pixel, cùng α; MultinomialNB trên pixel [0, 1]."""
    from sklearn.naive_bayes import MultinomialNB

    feats = make_features(cfg["mask"], "none")
    return Pipeline(feats, MultinomialNB(alpha=cfg["alpha"])), f"MultinomialNB(alpha={cfg['alpha']:g})"


def build_mlp(cfg):
    """Cùng kiến trúc, SGD + momentum (không Nesterov), cùng γ, α, batch; huấn luyện đúng cfg["epochs"] epoch
    (epoch của trọng số mà bản tự cài giữ lại), không dừng sớm vì sklearn chỉ dừng sớm trên tập con của train.

    L2: sklearn cộng alpha·W / batch_size vào gradient của loss trung bình, bản tự cài cộng λW, nên alpha = λ·batch.
    Khác nhau còn lại: sklearn khởi tạo Glorot đều (bản tự cài: theo cfg["init"]) và không có dropout.
    """
    from sklearn.neural_network import MLPClassifier

    act = {"relu": "relu", "sigmoid": "logistic", "tanh": "tanh"}[cfg["activation"]]
    epochs = cfg.get("epochs", cfg["max_epochs"])
    model = MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]), activation=act, solver="sgd",
                          learning_rate="constant", learning_rate_init=cfg["lr"], momentum=cfg["momentum"],
                          nesterovs_momentum=False, batch_size=cfg["batch_size"],
                          alpha=cfg["weight_decay"] * cfg["batch_size"], max_iter=epochs, n_iter_no_change=epochs + 1,
                          tol=0.0, shuffle=True, random_state=SEED)
    note = f", {epochs} epoch" + (f" (bản tự cài có dropout {cfg['p_drop']}, sklearn không có)" if cfg["p_drop"] else "")
    return Pipeline(make_features("none", "scalar"), model), f"MLPClassifier(sgd{note})"


def build_logistic_regression_converged(cfg, n_train):
    """Cùng hàm mục tiêu với bản tự cài, giải tới hội tụ bằng lbfgs.

    Bản tự cài cực tiểu J(W) = CE trung bình + (λ/2)‖W‖²; LogisticRegression cực tiểu C·Σ CE + ½‖W‖², chia cho C·N
    thì trùng J khi C = 1/(λN). Cả hai không phạt bias.
    """
    from sklearn.linear_model import LogisticRegression

    C = 1.0 / (cfg["lam"] * n_train)
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-6)
    feats = make_features(cfg["mask"], cfg.get("scale", "scalar"))
    return Pipeline(feats, model), f"LogisticRegression(lbfgs tới hội tụ, C = 1/(λN) = {C:.3g})"


BUILDERS = {
    "naive_bayes": build_naive_bayes,
    "logistic_regression": build_logistic_regression,
    "mlp": build_mlp,
}


def build_alternatives(rf_cfg):
    """AdaBoost, Gradient Boosting trên cùng mã bin và cùng số cây với Random Forest cuối; siêu tham số cố định.

    AdaBoost (SAMME) dùng cây độ sâu 3: stump độ sâu 1 quá yếu cho 10 lớp. Gradient Boosting dùng bản histogram
    của sklearn: GradientBoostingClassifier cổ điển học 10 cây hồi quy mỗi vòng, tuần tự, quá chậm ở quy mô này.
    """
    from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    n = rf_cfg["n_estimators"]
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n, learning_rate=0.5, random_state=SEED)
    gb = HistGradientBoostingClassifier(max_iter=n, learning_rate=0.1, early_stopping=False, random_state=SEED)
    # (Pipeline, mô tả, số luồng huấn luyện): AdaBoost tuần tự; HistGradientBoosting dùng mọi lõi (OpenMP)
    return {
        "adaboost": (Pipeline(TreeFeatures(mask=rf_cfg["mask"], n_bins=rf_cfg["n_bins"]), ada),
                     f"AdaBoost (sklearn; SAMME, cây sâu 3, {n} cây, lr 0,5)", 1),
        "gradient_boosting": (Pipeline(TreeFeatures(mask=rf_cfg["mask"], n_bins=rf_cfg["n_bins"]), gb),
                              f"HistGradientBoosting (sklearn; {n} vòng, lr 0,1)", os.cpu_count()),
    }


def run_alternatives(splits, log):
    """In và ghi bảng Random Forest cuối so với AdaBoost, Gradient Boosting; None nếu chưa có kết quả RF.

    Cả ba đều là sklearn trên cùng đặc trưng, nên thời gian so được trực tiếp (ghi kèm số luồng).
    """
    rf = load_final_metrics("random_forest")
    if rf is None:
        log("\nPhương án ensemble bị loại: chưa có results/random_forest/final_metrics.json, bỏ qua")
        return None
    log(f"\nPhương án ensemble bị loại (EDA 5.3): cùng đặc trưng với RF ({rf['config']['mask']}, "
        f"{rf['config']['n_bins']} bin), siêu tham số cố định, không tinh chỉnh.")
    log(f"{'Mô hình':<52}{'val acc':>9}{'test acc':>10}{'test F1':>9}{'fit (s)':>9}{'pred (s)':>9}  luồng")
    out = {"random_forest": {"model": "Random Forest (sklearn, cấu hình cuối)", "fit_time": rf["fit_time"],
                             "n_jobs": rf["runtime"]["n_jobs"],
                             **{s: {k: rf["splits"][s][k] for k in ("accuracy", "f1_macro", "predict_time")}
                                for s in ("val", "test")}}}
    for key, (pipe, desc, n_threads) in build_alternatives(rf["config"]).items():
        res = evaluate(pipe, splits)
        out[key] = {"model": desc, "fit_time": res["fit_time"], "n_jobs": n_threads,
                    **{s: {k: res[s][k] for k in ("accuracy", "f1_macro", "predict_time")} for s in ("val", "test")}}
    for r in out.values():
        log(f"{r['model']:<52}{r['val']['accuracy']:>9.4f}{r['test']['accuracy']:>10.4f}{r['test']['f1_macro']:>9.4f}"
            f"{r['fit_time']:>9.1f}{r['test']['predict_time']:>9.3f}  {r['n_jobs']}")
    path = write_json(out, model_dir(OUT_KEY) / "ensemble_alternatives.json")
    log(f"Đã ghi {rel_path(path)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=[k for k in MODEL_KEYS if k in BUILDERS])
    ap.add_argument("--no-alternatives", action="store_true", help="không chạy AdaBoost, Gradient Boosting")
    args = ap.parse_args()

    log = Logger(model_dir(OUT_KEY) / "log.txt")
    splits = load_splits()
    n_train = len(splits["train"][1])
    summary_path = model_dir(OUT_KEY) / "summary.json"
    out = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    out = {k: v for k, v in out.items() if k in BUILDERS}       # chỉ mô hình có bản sklearn tương đương
    log(f"{'Mô hình':<22}{'':>8}{'val acc':>9}{'test acc':>10}{'test F1':>9}{'fit (s)':>9}{'pred (s)':>9}  n_jobs")
    for key in args.models:
        ours = load_final_metrics(key)
        if ours is None:
            log(f"{key:<22}chưa có final_metrics.json, bỏ qua")
            continue
        if key not in BUILDERS:
            log(f"{key:<22}không có trong BUILDERS (không có bản tự cài để đối chiếu), bỏ qua")
            continue
        n_jobs = ours.get("runtime", {}).get("n_jobs", 1)
        cfg = ours["config"]
        if "epochs" not in cfg and "best_epoch" in (ours.get("extra") or {}):
            cfg = {**cfg, "epochs": ours["extra"]["best_epoch"]}   # LR: epoch giữ lại khi dừng sớm (như MLP)
        pipe, desc = BUILDERS[key](cfg)
        res = evaluate(pipe, splits)
        row = {"sklearn_model": desc, "n_jobs": n_jobs, "fit_time": res["fit_time"], "ours_fit_time": ours["fit_time"],
               "ours_quick": ours.get("quick", False)}
        for s in ("val", "test"):
            row[s] = {"accuracy": res[s]["accuracy"], "f1_macro": res[s]["f1_macro"],
                      "predict_time": res[s]["predict_time"],
                      "ours_accuracy": ours["splits"][s]["accuracy"], "ours_f1_macro": ours["splits"][s]["f1_macro"],
                      "ours_predict_time": ours["splits"][s].get("predict_time")}
        out[key] = row
        name = DISPLAY_NAMES.get(key, key)
        for who, v_acc, t_acc, t_f1, fit, pred in [
                ("tự cài", row["val"]["ours_accuracy"], row["test"]["ours_accuracy"], row["test"]["ours_f1_macro"],
                 row["ours_fit_time"], row["test"]["ours_predict_time"]),
                ("sklearn", row["val"]["accuracy"], row["test"]["accuracy"], row["test"]["f1_macro"],
                 row["fit_time"], row["test"]["predict_time"])]:
            log(f"{name if who == 'tự cài' else '':<22}{who:>8}{v_acc:>9.4f}{t_acc:>10.4f}{t_f1:>9.4f}"
                f"{fit or float('nan'):>9.1f}{pred or float('nan'):>9.3f}  {n_jobs}")
        if key == "logistic_regression":
            pipe_c, desc_c = build_logistic_regression_converged(cfg, n_train)
            res_c = evaluate(pipe_c, splits)
            row["converged"] = {"sklearn_model": desc_c, "fit_time": res_c["fit_time"],
                                **{s: {"accuracy": res_c[s]["accuracy"], "f1_macro": res_c[s]["f1_macro"]}
                                   for s in ("val", "test")}}
            log(f"{'':<22}{'lbfgs':>8}{res_c['val']['accuracy']:>9.4f}{res_c['test']['accuracy']:>10.4f}"
                f"{res_c['test']['f1_macro']:>9.4f}{res_c['fit_time']:>9.1f}{res_c['test']['predict_time']:>9.3f}"
                f"  {n_jobs}   ({desc_c})")
        if ours.get("quick"):
            log(f"{'':<22}CẢNH BÁO: kết quả tự cài là bản --quick")
    path = write_json(out, summary_path)
    log(f"Đã ghi {rel_path(path)} ({', '.join(out) or 'trống'})")
    if not args.no_alternatives:
        run_alternatives(splits, log)


if __name__ == "__main__":
    main()
