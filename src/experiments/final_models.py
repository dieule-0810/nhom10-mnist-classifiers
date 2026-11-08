"""Mô hình cuối: huấn luyện lại từ cấu hình đã chốt, lưu ra đĩa, nạp lại và kiểm tra tái lập.

Các script thí nghiệm dò siêu tham số rồi ghi cấu hình cuối vào results/<key>/final_metrics.json (vài giờ).
File này chỉ huấn luyện MÔ HÌNH CUỐI theo đúng cấu hình đó (khoảng 35–40 phút cho mọi mô hình), lưu vào
results/<key>/model.pkl (5 CNN của tổ hợp: results/ensemble/cnn_s<seed>.pkl), rồi dự đoán lại trên train / val /
test và so ma trận nhầm lẫn với final_metrics.json: khớp tức là kết quả trong báo cáo tái lập được.
notebooks/results.ipynb dùng get_model / ensemble_proba ở đây; lưu / nạp file .pkl ở src/evaluation/results_io.py.

Chạy từ gốc repo, sau khi đã có final_metrics.json của các mô hình:
    python -m src.experiments.final_models                    # mọi mô hình và gộp mô hình
    python -m src.experiments.final_models --models cnn mlp

Tái lập cần cùng seed (src/config.py) VÀ cùng số luồng BLAS (--blas-threads, mặc định 8 như các script thí
nghiệm): đổi số luồng làm đổi thứ tự cộng số thực, nên MLP / CNN huấn luyện ra trọng số hơi khác.
Random Forest (hơn 100 MB, quá giới hạn của GitHub) không đưa lên Git: lần đầu gọi get_model sẽ tự huấn luyện lại
(vài giây); với random_state cố định, sklearn cho cùng một rừng dù chạy bao nhiêu luồng.
"""

import argparse
import time

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import BASELINE_KEY, DISPLAY_NAMES, ENSEMBLE_KEY, MODEL_KEYS, SEED, SPLITS
from src.data.preprocess import load_splits
from src.evaluation.metrics import classification_scores
from src.evaluation.results_io import RESULTS_DIR, load_final_metrics, load_model, model_path, save_model
from src.models.nearest_centroid import NearestCentroidNP

BLAS_THREADS = 8
KEYS = (BASELINE_KEY, *MODEL_KEYS)


# ---------------------------------------------------------------- dựng, huấn luyện
def build(key, cfg, cnn_seed=SEED):
    """Pipeline chưa huấn luyện của mô hình cuối, từ "config" trong final_metrics.json."""
    if key == BASELINE_KEY:
        return NearestCentroidNP()
    if key == "cnn":
        from src.experiments.cnn import make_cnn

        return make_cnn(dict(cfg, channels=tuple(cfg["channels"])), seed=cnn_seed)
    from src.experiments.ensemble import BUILDERS             # cùng hàm dựng mà ensemble dùng

    return BUILDERS[key](cfg, n_jobs=-1)


def fit(model, splits):
    """Huấn luyện như script thí nghiệm: trên train, val chỉ cho mô hình có dừng sớm."""
    X_tr, y_tr = splits["train"]
    t0 = time.perf_counter()
    if hasattr(model, "features"):                            # Pipeline: chuyển val nếu mô hình nhận
        model.fit(X_tr, y_tr, *splits["val"])
    else:
        model.fit(X_tr, y_tr)
    return model, time.perf_counter() - t0


def get_model(key, splits, retrain=False, log=print):
    """Nạp results/<key>/model.pkl; huấn luyện lại (và lưu) nếu retrain=True hoặc chưa có file."""
    if not retrain and model_path(key).is_file():
        return load_model(key)
    fm = load_final_metrics(key)
    if fm is None:
        raise FileNotFoundError(f"Chưa có results/{key}/final_metrics.json: chạy script thí nghiệm của {key} trước.")
    model, sec = fit(build(key, fm["config"]), splits)
    save_model(key, model)
    log(f"{DISPLAY_NAMES[key]}: huấn luyện lại {sec:.0f} s, đã lưu {model_path(key).relative_to(RESULTS_DIR.parent)}")
    return model


def ensemble_proba(splits, retrain=False, log=print):
    """Xác suất của phương án gộp đã chọn (config trong results/ensemble/final_metrics.json) trên mọi tập.

    Thành viên cổ điển dùng lại mô hình cuối của chính nó; mỗi CNN của seed ensemble lưu riêng
    (results/ensemble/cnn_s<seed>.pkl). Cách gộp giống hệt src/experiments/ensemble.py (soft voting có trọng số).
    """
    from src.experiments.ensemble import soft_vote

    cfg = load_final_metrics(ENSEMBLE_KEY)["config"]
    cnn_cfg = load_final_metrics("cnn")["config"]
    cache = {}

    def proba(name, split):
        if name not in cache:
            if name.startswith("CNN-s"):
                file = f"cnn_s{name[5:]}"
                if retrain or not model_path(ENSEMBLE_KEY, file).is_file():
                    model, sec = fit(build("cnn", cnn_cfg, cnn_seed=int(name[5:])), splits)
                    save_model(ENSEMBLE_KEY, model, file)
                    log(f"Gộp mô hình, {name}: huấn luyện lại {sec:.0f} s")
                cache[name] = load_model(ENSEMBLE_KEY, file)
            else:
                cache[name] = get_model(name, splits, retrain, log)
        return cache[name].predict_proba(splits[split][0])

    out = {}
    for s in SPLITS:
        Ps = [soft_vote([proba(c, s) for c in cfg["cnn_seeds"]]) if m == "CNN_ENS" else proba(m, s)
              for m in cfg["members"]]
        out[s] = soft_vote(Ps, cfg["weights"])
    return out


# ---------------------------------------------------------------- kiểm tra tái lập
def check(key, preds, splits):
    """So ma trận nhầm lẫn tính lại với final_metrics.json. Trả về {tập: (accuracy tính lại, đã ghi, khớp)}."""
    fm = load_final_metrics(key)
    out = {}
    for s in SPLITS:
        cm = classification_scores(splits[s][1], preds[s])["confusion_matrix"]
        rec = np.asarray(fm["splits"][s]["confusion_matrix"])
        out[s] = (float(np.trace(cm) / cm.sum()), fm["splits"][s]["accuracy"], bool(np.array_equal(cm, rec)))
    return out


def predictions(key, splits, retrain=False, log=print):
    if key == ENSEMBLE_KEY:
        return {s: P.argmax(axis=1) for s, P in ensemble_proba(splits, retrain, log).items()}
    model = get_model(key, splits, retrain, log)
    return {s: np.asarray(model.predict(splits[s][0])) for s in SPLITS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[*KEYS, ENSEMBLE_KEY])
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS, help="phải bằng số luồng lúc chạy thí nghiệm")
    ap.add_argument("--load", action="store_true", help="nạp mô hình đã lưu thay vì huấn luyện lại")
    args = ap.parse_args()

    splits = load_splits()
    ok = True
    with threadpool_limits(args.blas_threads, user_api="blas"):
        for key in args.models:
            res = check(key, predictions(key, splits, retrain=not args.load), splits)
            ok &= all(m for *_, m in res.values())
            print(f"{DISPLAY_NAMES[key]:<28}" + " | ".join(
                f"{s} {a:.4f} (ghi {r:.4f}) {'khớp' if m else 'LỆCH'}" for s, (a, r, m) in res.items()), flush=True)
    print("Mọi mô hình tái lập đúng ma trận nhầm lẫn đã ghi." if ok else "CÓ MÔ HÌNH LỆCH so với final_metrics.json.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
