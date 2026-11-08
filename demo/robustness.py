"""Đo độ bền của bước chuẩn hoá ảnh vẽ (demo/drawing.py) trên ảnh test MNIST vẽ lại to và lệch.

Mô phỏng người dùng vẽ to và không ở giữa: lấy N_IMAGES ảnh test ngẫu nhiên, phóng to mỗi ảnh SCALE lần
(28 × 7 = 196 pixel) rồi đặt vào khung 280×280 của demo, lệch khỏi tâm tới ±MAX_SHIFT pixel mỗi chiều.
Với mỗi mô hình của demo (demo/models.py: các results/<key>/model.pkl và CNN × 5), so bốn cách đưa vào mô hình:
    acc_original    ảnh 28×28 gốc (mốc)
    acc_pipeline    ảnh vẽ lại → to_mnist (cắt khung chữ, thu nhỏ vào 20×20, căn tâm khối lượng, bỏ quầng mờ)
    keep            tỉ lệ ảnh mà dự đoán qua to_mnist trùng với dự đoán trên ảnh gốc
    acc_naive       thu nhỏ thẳng cả khung 280×280 về 28×28 (không cắt, không căn tâm)
và thêm một biến thể to_mnist không bỏ quầng mờ (acc_no_halo), cùng số pixel có mực trung bình của hai cách.

Chạy từ gốc repo, SAU python -m src.experiments.final_models:
    python -m demo.robustness

Ghi results/demo/summary.json và log.txt; hình demo_pipeline.png do src/plots/demo.py vẽ (gọi ở cuối script).
"""

import numpy as np
from PIL import Image

from demo.drawing import CANVAS, SIZE, drawn_canvas, to_mnist
from demo.models import DEMO_MODELS as MODELS, demo_paths, is_available, load_demo_model
from src.config import DISPLAY_NAMES, SEED
from src.data.preprocess import load_splits
from src.evaluation.results_io import Logger, model_dir, rel_path, write_json
from src.plots.demo import KEY, plot_all
N_IMAGES = 300
SCALE = 7
MAX_SHIFT = 40
EXAMPLE_DIGIT, EXAMPLE_SHIFT = 7, (-30, 40)       # hình minh hoạ: một chữ số 7 vẽ to, lệch lên và lệch phải


def naive_resize(canvas):
    """Thu nhỏ thẳng cả khung vẽ về 28×28 (không cắt, không căn tâm): mực trong [0, 1], shape (784,)."""
    ink = (255 * (1 - canvas)).astype(np.uint8)
    small = Image.fromarray(ink).resize((SIZE, SIZE), Image.LANCZOS)
    return (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)


def main():
    log = Logger(model_dir(KEY) / "log.txt")
    X_te, y_te = load_splits()["test"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_te), N_IMAGES, replace=False)
    shifts = rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=(N_IMAGES, 2))
    canvases = [drawn_canvas(X_te[i], s, SCALE) for i, s in zip(idx, shifts)]
    X_orig, y = X_te[idx], y_te[idx]
    X_pipe = np.stack([to_mnist(c) for c in canvases])
    X_halo = np.stack([to_mnist(c, drop_halo=False) for c in canvases])
    X_naive = np.stack([naive_resize(c) for c in canvases])
    log(f"{N_IMAGES} ảnh test (seed {SEED}), phóng to {SCALE} lần, đặt vào khung {CANVAS}×{CANVAS} lệch tới "
        f"±{MAX_SHIFT} pixel")
    ink = dict(original=float((X_orig > 0).sum(1).mean()), pipeline=float((X_pipe > 0).sum(1).mean()),
               no_halo=float((X_halo > 0).sum(1).mean()))
    log(f"Số pixel có mực trung bình: ảnh gốc {ink['original']:.0f} | to_mnist {ink['pipeline']:.0f} | "
        f"to_mnist không bỏ quầng mờ {ink['no_halo']:.0f}")

    ex = int(next(i for i in range(len(X_te)) if y_te[i] == EXAMPLE_DIGIT))
    x_ex = to_mnist(drawn_canvas(X_te[ex], EXAMPLE_SHIFT, SCALE))[None, :]
    rows, example_proba = {}, {}
    log(f"\n{'Mô hình':<22}{'acc gốc':>9}{'to_mnist':>10}{'giữ':>7}{'thu nhỏ thẳng':>15}{'không bỏ quầng':>16}")
    for key in MODELS:
        if not is_available(key):
            log(f"{DISPLAY_NAMES[key]:<22}chưa có {', '.join(rel_path(p) for p in demo_paths(key))}, bỏ qua")
            continue
        model = load_demo_model(key)
        p_orig, p_pipe = model.predict(X_orig), model.predict(X_pipe)
        rows[key] = dict(acc_original=float((p_orig == y).mean()), acc_pipeline=float((p_pipe == y).mean()),
                         keep=float((p_pipe == p_orig).mean()), acc_naive=float((model.predict(X_naive) == y).mean()),
                         acc_no_halo=float((model.predict(X_halo) == y).mean()))
        example_proba[key] = model.predict_proba(x_ex)[0]
        r = rows[key]
        log(f"{DISPLAY_NAMES[key]:<22}{r['acc_original']:>9.3f}{r['acc_pipeline']:>10.3f}{r['keep']:>7.3f}"
            f"{r['acc_naive']:>15.3f}{r['acc_no_halo']:>16.3f}")

    summary = dict(n_images=N_IMAGES, seed=SEED, scale=SCALE, max_shift=MAX_SHIFT, ink_pixels=ink, models=rows,
                   example=dict(test_index=ex, label=EXAMPLE_DIGIT, shift=EXAMPLE_SHIFT, proba=example_proba))
    path = write_json(summary, model_dir(KEY) / "summary.json")
    plot_all()
    log(f"\nĐã ghi {rel_path(path)}, log.txt và figures/")


if __name__ == "__main__":
    main()
