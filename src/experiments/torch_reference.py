"""Đối chiếu CNN tự cài (NumPy) với cùng kiến trúc viết bằng PyTorch, trên cùng splits và cùng siêu tham số.

Vai trò như sklearn_reference (sklearn không có CNN): tests/check_cnn.py đã đối chiếu từng lớp với torch;
script này so ở quy mô thật: accuracy, F1, số epoch, thời gian huấn luyện và dự đoán.
PyTorch chỉ dùng ở đây và trong tests/, không dùng trong src/models/.

Chạy từ gốc repo, SAU python -m src.experiments.cnn:
    python -m src.experiments.torch_reference              # torch dùng đúng số luồng BLAS của CNN NumPy
    python -m src.experiments.torch_reference --threads 4  # đặt số luồng khác

So thời gian chỉ có nghĩa khi hai bản cùng số luồng: mặc định lấy runtime.n_jobs trong results/cnn/final_metrics.json.

Ghi results/torch_reference/summary.json và log.txt.
"""

import argparse
import time

import numpy as np

from src.config import SEED
from src.data.features import make_features
from src.data.preprocess import load_splits
from src.evaluation.metrics import classification_scores
from src.evaluation.results_io import Logger, load_final_metrics, model_dir, rel_path, write_json

OUT_KEY = "torch_reference"


def build_torch_cnn(cfg):
    """Cùng kiến trúc với src.models.cnn.build_layers: (conv 3×3 → ReLU → max-pool 2×2) × khối → 2 lớp đầy đủ."""
    import torch.nn as nn

    layers, c_in, size = [], 1, 28
    for c in cfg["channels"]:
        layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        c_in, size = c, size // 2
    layers += [nn.Flatten(), nn.Linear(c_in * size * size, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 10)]
    return nn.Sequential(*layers)


def fit_torch(cfg, X, y, X_val, y_val, seed=SEED):
    """Mini-batch Adam, dừng sớm theo val loss và khôi phục epoch tốt nhất, như CNNClassifierNP.fit."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    model = build_torch_cnn(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    xt = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)
    yt = torch.tensor(y)
    xv = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 1, 28, 28)
    yv = torch.tensor(y_val)
    g = torch.Generator().manual_seed(seed)
    best, best_loss, wait, history = None, np.inf, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for idx in torch.randperm(len(xt), generator=g).split(cfg["batch_size"]):
            opt.zero_grad()
            F.cross_entropy(model(xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(xv), yv))
        history.append(val_loss)
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best = ({k: v.clone() for k, v in model.state_dict().items()}, epoch)
        else:
            wait += 1
            if cfg["patience"] is not None and wait >= cfg["patience"]:
                break
    model.load_state_dict(best[0])
    return model, best[1], len(history)


def predict_torch(model, X, batch=1000):
    import torch

    x = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)
    with torch.no_grad():
        return np.concatenate([model(x[i:i + batch]).argmax(1).numpy() for i in range(0, len(x), batch)])


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=None, help="số luồng của torch (mặc định: bằng CNN NumPy)")
    args = ap.parse_args()
    log = Logger(model_dir(OUT_KEY) / "log.txt")
    ours = load_final_metrics("cnn")
    if ours is None:
        log("Chưa có results/cnn/final_metrics.json: chạy python -m src.experiments.cnn trước.")
        return
    if ours["quick"]:
        log("CẢNH BÁO: results/cnn là kết quả --quick.")
    cfg = ours["config"]
    threads = args.threads or ours.get("runtime", {}).get("n_jobs") or torch.get_num_threads()
    torch.set_num_threads(threads)
    splits = load_splits()
    feats = make_features("none", cfg["scale"])
    X_tr = feats.fit_transform(splits["train"][0])
    X_va = feats.transform(splits["val"][0])
    log(f"Cấu hình (results/cnn/final_metrics.json): {cfg} | torch {torch.__version__}, "
        f"{torch.get_num_threads()} luồng")

    t0 = time.perf_counter()
    model, best_epoch, epochs = fit_torch(cfg, X_tr, splits["train"][1], X_va, splits["val"][1])
    fit_s = time.perf_counter() - t0
    out = dict(config=cfg, torch_version=torch.__version__, threads=threads, fit_time=fit_s, best_epoch=best_epoch,
               epochs_run=epochs, ours=dict(fit_time=ours["fit_time"], best_epoch=ours["extra"]["best_epoch"],
                                            epochs_run=ours["extra"]["epochs_run"]))
    log(f"\n{'':<6}{'acc NumPy':>11}{'acc torch':>11}{'chênh':>9}{'F1 NumPy':>10}{'F1 torch':>10}"
        f"{'dự đoán NumPy':>15}{'dự đoán torch':>15}")
    for name in ("train", "val", "test"):
        X, y = splits[name]
        t0 = time.perf_counter()
        pred = predict_torch(model, feats.transform(X))
        pred_s = time.perf_counter() - t0
        s = classification_scores(y, pred)
        o = ours["splits"][name]
        out[name] = dict(accuracy=s["accuracy"], f1_macro=s["f1_macro"], predict_time=pred_s,
                         ours_accuracy=o["accuracy"], ours_f1_macro=o["f1_macro"], ours_predict_time=o["predict_time"])
        log(f"{name:<6}{o['accuracy']:>11.4f}{s['accuracy']:>11.4f}{o['accuracy'] - s['accuracy']:>+9.4f}"
            f"{o['f1_macro']:>10.4f}{s['f1_macro']:>10.4f}{o['predict_time']:>14.2f}s{pred_s:>14.2f}s")
    log(f"\nHuấn luyện: NumPy {ours['fit_time']:.0f} s ({out['ours']['epochs_run']} epoch, giữ epoch "
        f"{out['ours']['best_epoch']}) | torch {fit_s:.0f} s ({epochs} epoch, giữ epoch {best_epoch}) | "
        f"NumPy chậm hơn × {ours['fit_time'] / fit_s:.1f}")
    log("Hai bản khác khởi tạo (He chuẩn so với Kaiming đều của torch) và khác dãy ngẫu nhiên trộn mini-batch, "
        "nên chỉ kỳ vọng accuracy gần nhau (cỡ độ lệch do seed), không trùng từng ảnh.")
    path = write_json(out, model_dir(OUT_KEY) / "summary.json")
    log(f"\nĐã ghi {rel_path(path)} và log.txt")


if __name__ == "__main__":
    main()
