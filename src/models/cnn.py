"""Mạng nơron tích chập (CNN) tự cài bằng NumPy: conv → ReLU → max-pool → ... → softmax.

Kiến trúc mặc định giống ví dụ PyTorch trong slide "Ensemble cho ảnh MNIST" (bài 06, slide 12):
    Conv 3×3 (1 → 16, padding 1) → ReLU → MaxPool 2×2      28×28 → 14×14
    Conv 3×3 (16 → 32, padding 1) → ReLU → MaxPool 2×2     14×14 → 7×7
    Flatten → Dense 1568 → 128 → ReLU → Dense 128 → 10 → softmax

Bố cục dữ liệu bên trong là NHWC (N, cao, rộng, kênh) để phép im2col chỉ cần reshape, không chuyển trục.
Trọng số conv lưu (F, C, kh, kw) như PyTorch, nên tests/ chép thẳng sang torch để đối chiếu.

Mô hình nhận X phẳng (N, 784) như mọi bộ phân lớp của nhóm (src/models/base.py), tự reshape về
(N, 28, 28, 1); vì vậy đặc trưng đi trước KHÔNG được bỏ pixel (make_features(mask="none", ...)).

Tham khảo:
    [1] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to document
        recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
    [2] D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," ICLR, 2015.
    [3] K. He et al., "Delving deep into rectifiers," ICCV, 2015 (khởi tạo He).
    [4] I. Goodfellow, Y. Bengio, A. Courville, Deep Learning. MIT Press, 2016. Chương 9.
"""

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import IMAGE_SHAPE, N_CLASSES, SEED
from src.models._numeric import log_softmax
from src.models.base import BaseClassifier


# ---------------------------------------------------------------------------
# Các lớp. Mỗi lớp: forward(x) lưu cache, backward(d) trả về gradient theo đầu vào
# và ghi gradient tham số vào self.grads (cùng thứ tự với self.params).
# ---------------------------------------------------------------------------
class Conv2D:
    """Tích chập 2 chiều stride 1, padding đối xứng, bằng im2col.

    x (N, H, W, C) → cửa sổ (N, H, W, C, k, k) → ma trận (N·H·W, C·k·k) @ Wᵀ (C·k·k, F).
    Thứ tự (C, kh, kw) trong một hàng của im2col trùng với W.reshape(F, -1).
    """

    def __init__(self, c_in, c_out, k=3, padding=1, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng(SEED)
        fan_in = c_in * k * k
        self.W = (rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2.0 / fan_in)).astype(dtype)
        self.b = np.zeros(c_out, dtype=dtype)
        self.k, self.padding = k, padding
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x, train=True):
        p, k = self.padding, self.k
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p else x
        N, Hp, Wp, C = xp.shape
        H, W = Hp - k + 1, Wp - k + 1
        cols = sliding_window_view(xp, (k, k), axis=(1, 2)).reshape(N * H * W, C * k * k)
        out = cols @ self.W.reshape(len(self.W), -1).T + self.b
        if train:
            self._cache = (cols, xp.shape)
        return out.reshape(N, H, W, -1)

    def backward(self, d, need_dx=True):
        cols, (N, Hp, Wp, C) = self._cache
        k, p = self.k, self.padding
        F = len(self.W)
        d2 = d.reshape(-1, F)
        # (colsᵀ d)ᵀ thay vì dᵀ cols: cùng kết quả, nhưng dᵀ cols (F × N·H·W nhân N·H·W × C·k·k) chậm
        # hơn ~500 lần trên BLAS của NumPy (đo: 2 s so với 4 ms mỗi batch ở conv thứ hai)
        self.grads[0][...] = (cols.T @ d2).T.reshape(self.W.shape)
        self.grads[1][...] = d2.sum(axis=0)
        if not need_dx:
            return None
        H, W = Hp - k + 1, Wp - k + 1
        dcols = (d2 @ self.W.reshape(F, -1)).reshape(N, H, W, C, k, k)
        dxp = np.zeros((N, Hp, Wp, C), dtype=d.dtype)
        for i in range(k):                                  # col2im: cộng dồn k·k lát cắt
            for j in range(k):
                dxp[:, i:i + H, j:j + W, :] += dcols[..., i, j]
        return dxp[:, p:Hp - p, p:Wp - p, :] if p else dxp


class ReLU:
    params, grads = [], []

    def forward(self, x, train=True):
        if train:
            self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, d):
        return d * self._mask


class MaxPool2x2:
    """Max-pool 2×2 stride 2 (cửa sổ không chồng lấp). Gradient chỉ về MỘT vị trí lớn nhất mỗi cửa sổ
    (vị trí đầu tiên nếu bằng nhau), giống PyTorch."""

    params, grads = [], []

    def forward(self, x, train=True):
        N, H, W, C = x.shape
        win = x.reshape(N, H // 2, 2, W // 2, 2, C).transpose(0, 1, 3, 5, 2, 4).reshape(N, H // 2, W // 2, C, 4)
        idx = win.argmax(axis=-1)
        if train:
            self._cache = (idx, x.shape)
        return np.take_along_axis(win, idx[..., None], axis=-1)[..., 0]

    def backward(self, d):
        idx, (N, H, W, C) = self._cache
        dwin = np.zeros((*d.shape, 4), dtype=d.dtype)
        np.put_along_axis(dwin, idx[..., None], d[..., None], axis=-1)
        return dwin.reshape(N, H // 2, W // 2, C, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(N, H, W, C)


class Flatten:
    params, grads = [], []

    def forward(self, x, train=True):
        if train:
            self._shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, d):
        return d.reshape(self._shape)


class Dense:
    def __init__(self, d_in, d_out, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng(SEED)
        self.W = (rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)).astype(dtype)
        self.b = np.zeros(d_out, dtype=dtype)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x, train=True):
        if train:
            self._x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.grads[0][...] = self._x.T @ d
        self.grads[1][...] = d.sum(axis=0)
        return d @ self.W.T


class Adam:
    """Adam [2] với hiệu chỉnh độ lệch; cập nhật tại chỗ các mảng tham số."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr, self.beta1, self.beta2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.beta1 ** self.t
        c2 = 1 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def build_layers(channels=(16, 32), hidden=128, rng=None, dtype=np.float32, image_shape=IMAGE_SHAPE,
                 n_classes=N_CLASSES):
    """Dựng danh sách lớp. Mỗi khối (conv → ReLU → pool) giảm kích thước ảnh một nửa."""
    rng = rng or np.random.default_rng(SEED)
    layers, c_in = [], 1
    h, w = image_shape
    for c in channels:
        layers += [Conv2D(c_in, c, rng=rng, dtype=dtype), ReLU(), MaxPool2x2()]
        c_in, h, w = c, h // 2, w // 2
    layers += [Flatten(), Dense(h * w * c_in, hidden, rng=rng, dtype=dtype), ReLU(),
               Dense(hidden, n_classes, rng=rng, dtype=dtype)]
    return layers


# ---------------------------------------------------------------------------
# Bộ phân lớp
# ---------------------------------------------------------------------------
class CNNClassifierNP(BaseClassifier):
    """CNN huấn luyện bằng mini-batch Adam, hàm mất mát cross-entropy, dừng sớm theo val loss.

    Siêu tham số:
        channels    số bộ lọc của từng khối conv; mỗi khối giảm ảnh một nửa (28 → 14 → 7)
        hidden      số nơron lớp ẩn đầy đủ
        lr          learning rate của Adam
        batch_size  kích thước mini-batch
        max_epochs  số epoch tối đa
        patience    dừng khi val loss không giảm sau ngần ấy epoch (chỉ khi fit có X_val); None = chạy hết
        seed        khởi tạo trọng số và thứ tự trộn mini-batch (hai nguồn đa dạng của seed ensemble)

    Sau fit: history_ (mỗi epoch: train_loss, train_acc trung bình trong epoch, val_loss, val_acc, giây),
    best_epoch_ (epoch có val loss nhỏ nhất, trọng số được khôi phục về epoch này), n_params_.
    """

    def __init__(self, channels=(16, 32), hidden=128, lr=1e-3, batch_size=128, max_epochs=10, patience=3,
                 seed=SEED, verbose=False):
        self.channels = tuple(channels)
        self.hidden = hidden
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

    # --- tiện ích ---
    @staticmethod
    def _images(X):
        return np.ascontiguousarray(X, dtype=np.float32).reshape(len(X), *IMAGE_SHAPE, 1)

    def _params(self):
        return [p for layer in self.layers_ for p in layer.params]

    def _grads(self):
        return [g for layer in self.layers_ for g in layer.grads]

    def _forward(self, x, train):
        for layer in self.layers_:
            x = layer.forward(x, train)
        return x

    def _backward(self, d):
        for layer in reversed(self.layers_[1:]):
            d = layer.backward(d)
        self.layers_[0].backward(d, need_dx=False)          # lớp đầu: không cần gradient theo ảnh

    def _loss_grad(self, logits, y):
        """Cross-entropy trung bình và gradient theo logits: (softmax − one-hot) / N."""
        logp = log_softmax(logits)
        n = len(y)
        loss = -float(logp[np.arange(n), y].mean())
        d = np.exp(logp)
        d[np.arange(n), y] -= 1
        return loss, (d / n).astype(logits.dtype), logp

    def _logits(self, X, batch=1000):
        x = self._images(X)
        return np.concatenate([self._forward(x[i:i + batch], train=False) for i in range(0, len(x), batch)])

    # --- interface chung ---
    def fit(self, X, y, X_val=None, y_val=None):
        X, y = self._validate_fit(X, y)
        rng = np.random.default_rng(self.seed)
        self.layers_ = build_layers(self.channels, self.hidden, rng=rng)
        self.n_params_ = int(sum(p.size for p in self._params()))
        opt = Adam(self._params(), lr=self.lr)
        x = self._images(X)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = self._validate_predict(X_val)
            y_val = np.asarray(y_val, dtype=np.int64)

        self.history_ = []
        best, best_loss, wait = None, np.inf, 0
        for epoch in range(1, self.max_epochs + 1):
            start = time.perf_counter()
            order = rng.permutation(len(x))
            loss_sum = correct = 0.0
            for i in range(0, len(x), self.batch_size):
                idx = order[i:i + self.batch_size]
                logits = self._forward(x[idx], train=True)
                loss, d, logp = self._loss_grad(logits, y[idx])
                self._backward(d)
                opt.step(self._grads())
                loss_sum += loss * len(idx)
                correct += float((logp.argmax(axis=1) == y[idx]).sum())
            row = {"epoch": epoch, "train_loss": loss_sum / len(x), "train_acc": correct / len(x)}
            if has_val:
                logp = log_softmax(self._logits(X_val))
                row["val_loss"] = -float(logp[np.arange(len(y_val)), y_val].mean())
                row["val_acc"] = float((logp.argmax(axis=1) == y_val).mean())
            row["seconds"] = time.perf_counter() - start
            self.history_.append(row)
            if self.verbose:
                print("  " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                        for k, v in row.items()), flush=True)
            if has_val:
                if row["val_loss"] < best_loss:
                    best_loss, wait = row["val_loss"], 0
                    best = ([p.copy() for p in self._params()], epoch)
                else:
                    wait += 1
                    if self.patience is not None and wait >= self.patience:
                        break
        if best is not None:
            for p, saved in zip(self._params(), best[0]):
                p[...] = saved
            self.best_epoch_ = best[1]
        else:
            self.best_epoch_ = len(self.history_)
        return self

    def decision_function(self, X):
        X = self._validate_predict(X)
        return self._logits(X).astype(np.float64)

    def predict_proba(self, X):
        return np.exp(log_softmax(self.decision_function(X)))

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)
