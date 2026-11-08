"""Biến đổi đặc trưng dùng chung cho mọi mô hình, chốt từ EDA (eda/notebooks/eda.ipynb).

preprocess.py chỉ làm các bước "cơ khí" ([0, 1], làm phẳng, chia tập). File này chứa các bước
mà EDA đã lập luận. Mọi tham số (mean, std, mặt nạ pixel, ngưỡng lượng tử) chỉ học từ
splits["train"] rồi áp nguyên cho val và test.

    PixelMask          : bỏ dead pixel / pixel hiếm / crop 24×24, 20×20   (EDA 1.1, 1.3)    mọi mô hình
    ScalarStandardizer : một mean, một std cho mọi pixel                  (EDA 1, 5.2)     LR, MLP
    PixelStandardizer  : mean, std riêng từng pixel - chỉ để đối chứng    (EDA 1)          LR (E1)
    Quantizer          : cường độ [0, 1] -> mã bin 0..B-1 (uint8)         (EDA 1.1, 5.3)   RF, NB
    FeaturePipeline    : nối các bước trên theo thứ tự
    make_features      : cấu hình chuẩn cho LR / MLP
    TreeFeatures       : PixelMask -> Quantizer, cấu hình chuẩn cho cây

Mỗi bước có fit / transform / fit_transform. Bước tuyến tính có thêm backproject(W): đưa trọng số
từ không gian đặc trưng về lưới 784 pixel (ảnh trọng số, giả thuyết H4).
"""

import numpy as np

from src.config import IMAGE_SHAPE, N_PIXELS, RARE_FREQ


# ---------------------------------------------------------------------------
# Chuẩn hoá
# ---------------------------------------------------------------------------
class ScalarStandardizer:
    """Chuẩn hoá vô hướng: x ← (x − μ) / σ, với μ, σ là MỘT số tính trên mọi pixel của tập train.

    Vì sao không chuẩn hoá từng pixel (EDA mục 1.1 và 5.2):
    72 dead pixel có σ = 0, còn 116 pixel gần như chết có σ rất nhỏ, nên z-score từng pixel
    khuếch đại vài chục ảnh hiếm thành giá trị rất lớn (nhiễu). Chuẩn hoá vô hướng giữ nguyên
    tỉ lệ giữa các pixel, chỉ đưa thang đo về quanh 0 để gradient descent ổn định.
    """

    def fit(self, X):
        self.mean_ = float(np.mean(X, dtype=np.float64))
        self.std_ = float(np.std(X, dtype=np.float64))
        if self.std_ == 0:
            self.std_ = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def backproject(self, W):
        """w · (x − μ)/σ = (w/σ) · x − w·μ/σ: trọng số theo pixel gốc là w/σ (phần hằng dồn vào bias)."""
        return W / self.std_

    def __repr__(self):
        return "ScalarStandardizer()"


class PixelStandardizer:
    """Chuẩn hoá z-score từng pixel (σ = 0 thay bằng 1).

    KHÔNG dùng cho mô hình chính; chỉ để thí nghiệm E1 cho thấy vì sao EDA loại cách này.
    """

    def fit(self, X):
        self.mean_ = X.mean(axis=0, dtype=np.float64)
        self.std_ = X.std(axis=0, dtype=np.float64)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def backproject(self, W):
        return W / self.std_[:, None]

    def __repr__(self):
        return "PixelStandardizer()"


# ---------------------------------------------------------------------------
# Chọn pixel
# ---------------------------------------------------------------------------
class PixelMask:
    """Giữ lại một tập con pixel; mặt nạ học từ tập train rồi áp nguyên cho val, test.

    mode:
        "none"   : giữ cả 784 pixel
        "dead"   : bỏ pixel không có mực ở ảnh train nào (72 vị trí, EDA 1.1)      -> 712
        "rare"   : bỏ thêm pixel có mực ở < min_freq ảnh (72 + 116 vị trí, EDA 1.1) -> 596
        "crop24" : bỏ hai vòng ngoài cùng, giữ khung 24×24 (99,76% mực, EDA 1.3)   -> 576
        "crop20" : bỏ bốn vòng ngoài cùng, giữ khung 20×20                         -> 400

    Thuộc tính sau fit: freq_ (tần suất có mực, 784), keep_ (mặt nạ bool, 784),
    index_ (cột gốc 0..783 của từng đặc trưng), n_features_.

    Ghi chú cho mô hình tuyến tính: sau chuẩn hoá vô hướng, dead pixel KHÔNG còn bằng 0 mà thành
    hằng số −μ/σ ≈ −0,42 ở mọi ảnh. Cột hằng này trùng hướng với cột bias nên trọng số của nó vẫn
    nhận gradient, thành một bias thứ hai bị L2 phạt. Bỏ dead pixel loại hẳn vấn đề này.
    Với cây: đặc trưng hằng không bao giờ được chọn để chia, bỏ đi chỉ để chạy nhanh hơn.
    """

    MODES = ("none", "dead", "rare", "crop24", "crop20")

    def __init__(self, mode="dead", min_freq=RARE_FREQ):
        if mode not in self.MODES:
            raise ValueError(f"mode phải thuộc {self.MODES}, nhận {mode!r}.")
        self.mode = mode
        self.min_freq = min_freq

    def fit(self, X):
        X = np.asarray(X)
        if X.shape[1] != N_PIXELS:
            raise ValueError(f"PixelMask cần đầu vào {N_PIXELS} pixel, nhận {X.shape[1]}.")
        freq = (X > 0).mean(axis=0)                       # tần suất có mực tại mỗi vị trí
        r, c = np.indices(IMAGE_SHAPE)
        ring = np.minimum.reduce([r, c, IMAGE_SHAPE[0] - 1 - r, IMAGE_SHAPE[1] - 1 - c]).ravel()  # 0 = viền
        if self.mode == "none":
            keep = np.ones(N_PIXELS, dtype=bool)
        elif self.mode == "dead":
            keep = freq > 0
        elif self.mode == "rare":
            keep = freq >= self.min_freq
        elif self.mode == "crop24":
            keep = ring >= 2
        else:                                             # crop20
            keep = ring >= 4
        self.freq_ = freq
        self.keep_ = keep
        self.index_ = np.flatnonzero(keep)
        self.n_features_ = int(keep.sum())
        return self

    def transform(self, X):
        # X[:, mask] trả về mảng xếp theo cột (F order); lấy mini-batch theo hàng trên mảng đó
        # chậm hơn khoảng 3 lần, nên ép về C order.
        return np.ascontiguousarray(np.asarray(X)[:, self.keep_])

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def backproject(self, W):
        """Đặt trọng số (d_giữ, ...) về đúng vị trí trên lưới 784; pixel bị bỏ mang giá trị NaN."""
        W = np.asarray(W)
        full = np.full((N_PIXELS,) + W.shape[1:], np.nan, dtype=np.float64)
        full[self.keep_] = W
        return full

    def to_image(self, values, fill=np.nan):
        """Một vectơ theo đặc trưng (độ dài = số pixel giữ lại) -> ảnh 28×28; pixel bị bỏ mang `fill`."""
        img = np.full(N_PIXELS, fill, dtype=np.float64)
        img[self.index_] = values
        return img.reshape(IMAGE_SHAPE)

    def __repr__(self):
        return f"PixelMask(mode={self.mode!r})"


# ---------------------------------------------------------------------------
# Nối các bước
# ---------------------------------------------------------------------------
class FeaturePipeline:
    """Chuỗi các bước biến đổi; fit lần lượt trên tập train."""

    def __init__(self, steps):
        self.steps = list(steps)

    def fit_transform(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return X

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def backproject(self, W):
        """Trọng số (d_đặc_trưng, K) -> (784, K) theo pixel gốc [0, 1]."""
        W = np.asarray(W, dtype=np.float64)
        for step in reversed(self.steps):
            W = step.backproject(W)
        return W

    def __repr__(self):
        return " -> ".join(repr(s) for s in self.steps) or "Identity"


def make_features(mask="dead", scale="scalar", min_freq=RARE_FREQ):
    """Dựng FeaturePipeline theo thứ tự: chọn pixel -> chuẩn hoá.

    Cấu hình mặc định là lựa chọn của EDA cho Logistic Regression và MLP (mục 5.2, 5.4):
    bỏ dead pixel, chuẩn hoá vô hướng.
    """
    steps = []
    if mask != "none":
        steps.append(PixelMask(mask, min_freq))
    scalers = {"scalar": ScalarStandardizer, "pixel": PixelStandardizer}
    if scale in scalers:
        steps.append(scalers[scale]())
    elif scale != "none":
        raise ValueError(f"scale phải thuộc {['none', *scalers]}, nhận {scale!r}.")
    return FeaturePipeline(steps)


# ---------------------------------------------------------------------------
# Lượng tử hoá (cây, Naive Bayes)
# ---------------------------------------------------------------------------
def to_uint8(X):
    """[0, 1] -> 0..255 (uint8). Dữ liệu sinh từ to_unit_range nên phép làm tròn là chính xác."""
    return np.rint(np.asarray(X, dtype=np.float32) * 255.0).astype(np.uint8)


class Quantizer:
    """Đổi cường độ pixel thành mã bin 0..B-1 (uint8), ngưỡng học từ tập train.

    Bin 0 luôn là "không có mực" (giá trị 0), vì 80,86% pixel bằng 0 (EDA 1.1). Các bin còn lại chia
    các giá trị khác 0 theo phân vị trên train, nên mỗi bin chứa xấp xỉ cùng số pixel khác 0.
    n_bins = 2 là nhị phân hoá (x > 0); n_bins >= 256 giữ nguyên 256 mức xám.

    Với cây: phép chia "x_j ≤ t" không đổi qua biến đổi đơn điệu, nên lượng tử chỉ giới hạn số
    ngưỡng ứng viên. Mã ≤ b tương ứng với x < cuts_[b] / 255 trên thang [0, 1].
    """

    def __init__(self, n_bins=16):
        if n_bins < 2:
            raise ValueError("n_bins phải ≥ 2")
        self.n_bins = n_bins

    def fit(self, X):
        v = to_uint8(X).ravel()
        if self.n_bins >= 256:
            cuts = np.arange(1, 256)
        else:
            nz = v[v > 0]
            q = np.quantile(nz, np.linspace(0, 1, self.n_bins)[1:-1], method="lower") if nz.size else []
            # Bin k (k ≥ 1) chứa các giá trị trong [cuts[k-1], cuts[k]); phân vị trùng nhau thì gộp lại
            cuts = np.unique(np.concatenate([[1], np.asarray(q, dtype=np.int64) + 1]))
            cuts = cuts[(cuts >= 1) & (cuts <= 255)]
        self.cuts_ = cuts.astype(np.int64)
        self.n_bins_ = len(self.cuts_) + 1
        self.lut_ = np.searchsorted(self.cuts_, np.arange(256), side="right").astype(np.uint8)  # uint8 -> mã bin
        return self

    def transform(self, X):
        return self.lut_[to_uint8(X)]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __repr__(self):
        return f"Quantizer(n_bins={self.n_bins})"


class TreeFeatures:
    """PixelMask rồi Quantizer. Đầu ra: ma trận mã bin uint8, shape (N, số pixel giữ lại)."""

    def __init__(self, mask="none", n_bins=16):
        self.mask = mask
        self.n_bins = n_bins

    def fit(self, X):
        self.mask_ = PixelMask(self.mask).fit(X)
        self.quantizer_ = Quantizer(self.n_bins).fit(self.mask_.transform(X))
        self.n_features_ = self.mask_.n_features_
        return self

    def transform(self, X):
        return self.quantizer_.transform(self.mask_.transform(X))

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_image(self, values, fill=np.nan):
        return self.mask_.to_image(values, fill)

    def __repr__(self):
        return f"TreeFeatures(mask={self.mask!r}, n_bins={self.n_bins})"
