"""Phân tích kết quả dùng chung cho mọi mô hình, để kiểm chứng giả thuyết EDA (eda/notebooks/eda.ipynb, mục 5.6).

Mọi script mô hình và compare_models.py dùng chung các hàm này, nên số liệu của các mô hình so được với nhau.
Mỗi khái niệm chỉ có MỘT hàm. Quy ước tham số giống sklearn: (y_true, y_pred | proba).

    Lỗi theo cặp lớp (H3)      : pair_errors, top_confused_pairs, share_in_pairs
    Xác suất (H1)              : log_loss, overconfident_error_rate, probabilistic_scores
    Giải thích mô hình (H4)    : fisher_scores, spearman, block_ids, block_sums, block_permutation_importance
    Phân tích lỗi              : confident_errors, confident_error_records

Tham khảo:
    [1] T. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning, 2nd ed.,
        Springer, 2009. Mục 15.3.2 (độ quan trọng hoán vị).
"""

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from src.config import BLOCK, EDA_CLOSE_PAIRS, IMAGE_SHAPE, N_CLASSES, RARE_FREQ

N_PAIRS = N_CLASSES * (N_CLASSES - 1) // 2          # 45 cặp; 5 cặp rải đều chiếm 5/45 ≈ 11,1% lỗi


# ---------------------------------------------------------------- lỗi theo cặp lớp
def pair_errors(cm, pairs=None):
    """{(i, j): cm[i, j] + cm[j, i]} (cộng hai chiều), i < j.

    pairs=None: đủ 45 cặp; truyền EDA_CLOSE_PAIRS để chỉ lấy 5 cặp EDA 3.1.
    Khi ghi JSON, results_io.to_jsonable đổi khoá (4, 9) thành chuỗi "(4, 9)".
    """
    cm = np.asarray(cm)
    pairs = combinations(range(len(cm)), 2) if pairs is None else pairs
    return {(int(i), int(j)): int(cm[i, j] + cm[j, i]) for i, j in pairs}


def top_confused_pairs(cm, k=5):
    """k cặp (i < j) nhầm nhiều nhất: list (i, j, số lỗi), giảm dần; hoà thì giữ thứ tự (i, j)."""
    pe = pair_errors(cm)
    order = sorted(pe, key=lambda p: -pe[p])            # sorted ổn định
    return [(i, j, pe[(i, j)]) for i, j in order[:k]]


def share_in_pairs(cm, pairs=EDA_CLOSE_PAIRS):
    """Tỉ lệ tổng số lỗi (ngoài đường chéo) rơi vào các cặp cho trước (H3)."""
    cm = np.asarray(cm)
    total = cm.sum() - np.trace(cm)
    return float(sum(pair_errors(cm, pairs).values()) / total) if total else 0.0


# ---------------------------------------------------------------- xác suất
def log_loss(y_true, proba, eps=1e-15):
    """−(1/N) Σ log p(y_i | x_i); p được kẹp vào [eps, 1] như sklearn để tránh log 0."""
    y_true = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(proba)[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.log(p).mean())


def overconfident_error_rate(y_true, proba, threshold=0.99):
    """Tỉ lệ mẫu (trên TOÀN tập) bị đoán sai mà xác suất của lớp được đoán > threshold (H1)."""
    proba = np.asarray(proba)
    wrong = proba.argmax(axis=1) != np.asarray(y_true)
    return float((wrong & (proba.max(axis=1) > threshold)).mean())


def probabilistic_scores(y_true, proba, threshold=0.99):
    """Mọi độ đo xác suất của một tập, đúng các khoá của final_metrics.json["splits"][s]["probabilistic"].

    So mean_confidence (xác suất lớn nhất, trung bình) với accuracy: lớn hơn là quá tự tin, nhỏ hơn là thiếu tự tin.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    wrong = proba.argmax(axis=1) != y_true
    over = overconfident_error_rate(y_true, proba, threshold)
    return {
        "log_loss": log_loss(y_true, proba),
        "overconfident_errors": over,
        "overconfident_share_of_errors": float(over * len(y_true) / wrong.sum()) if wrong.any() else 0.0,
        "mean_confidence": float(proba.max(axis=1).mean()),
        "mean_confidence_errors": float(proba.max(axis=1)[wrong].mean()) if wrong.any() else None,
    }


# ---------------------------------------------------------------- giải thích
def fisher_scores(X, y, min_freq=RARE_FREQ, n_classes=N_CLASSES):
    """Fisher score từng pixel (EDA 3.3): F_p = Σ_k n_k (μ_kp − μ_p)² / Σ_k n_k σ²_kp, trên ảnh [0, 1].

    Pixel có tần suất có mực < min_freq (gồm dead pixel) mang NaN, như EDA (ước lượng không tin cậy).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_k = np.bincount(y, minlength=n_classes).astype(np.float64)
    mu_k = np.stack([X[y == k].mean(axis=0) for k in range(n_classes)])
    var_k = np.stack([X[y == k].var(axis=0) for k in range(n_classes)])
    mu = (n_k[:, None] * mu_k).sum(axis=0) / n_k.sum()
    between = (n_k[:, None] * (mu_k - mu) ** 2).sum(axis=0)
    within = (n_k[:, None] * var_k).sum(axis=0)
    F = np.full(X.shape[1], np.nan)
    ok = ((X > 0).mean(axis=0) >= min_freq) & (within > 0)
    F[ok] = between[ok] / within[ok]
    return F


def spearman(a, b):
    """Tương quan hạng Spearman (scipy), bỏ vị trí có NaN/inf ở một trong hai; một vế hằng -> NaN."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def block_ids(block=BLOCK, shape=IMAGE_SHAPE):
    """Chỉ số khối (block × block) của từng pixel, đánh số theo hàng. 28×28 với block 4 -> 49 khối."""
    r, c = np.divmod(np.arange(shape[0] * shape[1]), shape[1])
    per_row = int(np.ceil(shape[1] / block))
    return (r // block) * per_row + (c // block)


def block_sums(values, block=BLOCK, nan_policy="omit"):
    """Cộng một bản đồ 784 pixel theo khối. Khối toàn NaN mang NaN."""
    values = np.asarray(values, dtype=np.float64).ravel()
    ids = block_ids(block)
    out = np.full(ids.max() + 1, np.nan)
    for g in range(len(out)):
        v = values[ids == g]
        if np.isfinite(v).any():
            out[g] = np.nansum(v) if nan_policy == "omit" else v.sum()
    return out


def block_permutation_importance(model, X, y, block=BLOCK, n_repeats=3, seed=0):
    """Độ quan trọng theo vùng, KHÔNG phụ thuộc mô hình: accuracy giảm bao nhiêu khi hoán vị một khối pixel.

    Cả khối dùng chung một hoán vị (theo ảnh), nên quan hệ giữa các pixel trong khối được giữ, chỉ mất
    quan hệ giữa khối đó và nhãn. Hoán vị từng pixel riêng lẻ sẽ đánh giá thấp độ quan trọng, vì pixel
    kề bên (r ≈ 0,71, EDA 2.1) vẫn mang cùng thông tin [1].

    model: bất kỳ đối tượng có predict nhận ảnh gốc (N, 784), ví dụ một Pipeline.
    Trả về mảng (số khối,) độ giảm accuracy trung bình; dùng chung cho mọi mô hình (H4, rubric 4).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    ids = block_ids(block)
    base = float((model.predict(X) == y).mean())
    drops = np.zeros(ids.max() + 1)
    for g in range(len(drops)):
        cols = np.flatnonzero(ids == g)
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops[g] += base - float((model.predict(Xp) == y).mean())
    return drops / n_repeats


# ---------------------------------------------------------------- phân tích lỗi
def confident_errors(y_true, proba, k=20):
    """Chỉ số k mẫu bị đoán sai có xác suất của lớp được đoán cao nhất (giảm dần)."""
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    wrong = np.flatnonzero(pred != np.asarray(y_true))
    return wrong[np.argsort(-proba[wrong, pred[wrong]], kind="stable")[:k]]


def confident_error_records(y_true, proba, k=20):
    """k lỗi tự tin nhất ở dạng lưu được vào JSON: {"idx": chỉ số ảnh, "proba": hàng xác suất (k, K)}.

    Đủ để vẽ lại eval_confident_errors.png (src/plots/common.py) mà không cần mô hình.
    """
    idx = confident_errors(y_true, proba, k)
    return {"idx": idx, "proba": np.asarray(proba)[idx]}
