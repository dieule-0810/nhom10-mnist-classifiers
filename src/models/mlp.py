"""Mạng nơ-ron nhiều lớp (MLP), huấn luyện bằng lan truyền ngược, tự cài bằng NumPy.

Gồm: khởi tạo (He, Xavier, hằng số), lan truyền thuận có
cache, BP1 đến BP4, đạo hàm ReLU / sigmoid / tanh, inverted dropout, số hạng L2, SGD + momentum,
early stopping. PyTorch chỉ dùng trong tests/check_mlp.py để đối chiếu gradient với autograd.

mini-batch N mẫu xếp theo hàng:
    Z^(l) = A^(l-1) (W^(l))^T + b^(l),   A^(l) = f(Z^(l)),   A^(0) = X
    W^(l) có shape (k_l, k_(l-1)) như nn.Linear; lớp ra không có hàm kích hoạt, softmax gộp vào hàm mất mát.

Tiền xử lý (EDA 5.4: chuẩn hoá vô hướng) KHÔNG nằm trong fit: ghép bằng
Pipeline(make_features("none", "scalar"), MLPClassifierNP(...)) như mọi mô hình khác (src/models/base.py).

Tham khảo:
    [1] D. E. Rumelhart, G. E. Hinton, R. J. Williams, "Learning representations by back-propagating errors,"
        Nature, vol. 323, pp. 533–536, 1986.
    [2] X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks,"
        AISTATS, 2010 (khởi tạo Xavier).
    [3] K. He et al., "Delving deep into rectifiers," ICCV, 2015 (khởi tạo He).
    [4] N. Srivastava et al., "Dropout: a simple way to prevent neural networks from overfitting," JMLR, 2014.
"""

import copy
import time

import numpy as np

from src.config import SEED
from src.models._numeric import log_softmax
from src.models.base import BaseClassifier


# ---------------------------------------------------------------------------
# Các lớp
class Linear:
    """z = a Wᵀ + b. Gradient sau backward nằm ở dW, db."""

    def __init__(self, k_in, k_out, init="he", const=0.01, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng(SEED)
        if init == "he":
            W = rng.standard_normal((k_out, k_in)) * np.sqrt(2.0 / k_in)          # std = sqrt(2 / k_in)
        elif init == "xavier":
            lim = np.sqrt(6.0 / (k_in + k_out))
            W = rng.uniform(-lim, lim, (k_out, k_in))                            # U(±sqrt(6 / (k_in + k_out)))
        elif init == "const":
            W = np.full((k_out, k_in), const)                                    # chỉ để minh hoạ đối xứng
        else:
            raise ValueError(f"init không hỗ trợ: {init}")
        self.W = W.astype(dtype)
        self.b = np.zeros(k_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)


class Activation:
    def __init__(self, name):
        if name not in ("relu", "sigmoid", "tanh"):
            raise ValueError(f"activation không hỗ trợ: {name}")
        self.name = name

    def __call__(self, z):
        if self.name == "relu":
            return np.maximum(z, 0)
        if self.name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        return np.tanh(z)


class Dropout:
    def __init__(self, p):
        self.p = p


def build_mlp(sizes, activation="relu", p_drop=0.0, init="he", const=0.01, seed=SEED, dtype=np.float32):
    """sizes = [784, k_1, ..., k_{L-1}, 10]. Trả về danh sách lớp:
    Linear -> Act -> (Dropout) -> ... -> Linear (lớp ra không có hàm kích hoạt).
    Cùng seed cho cùng trọng số khởi tạo (dùng lại ở phần giải thích W^(1))."""
    rng = np.random.default_rng(seed)
    layers = []
    L = len(sizes) - 1
    for l in range(L):
        layers.append(Linear(sizes[l], sizes[l + 1], init, const, rng, dtype))
        if l < L - 1:
            layers.append(Activation(activation))
            if p_drop > 0:
                layers.append(Dropout(p_drop))
    return layers


def linears(layers):
    return [layer for layer in layers if isinstance(layer, Linear)]


def n_params(layers):
    return int(sum(l.W.size + l.b.size for l in linears(layers)))


# ---------------------------------------------------------------------------
# Lan truyền thuận có cache
def forward_with_cache(layers, X, train=True, rng=None):
    """Trả về (output của mạng, cache). cache[i] = (layer, a_in, a_out, extra);
    extra là mặt nạ dropout khi huấn luyện, None trong mọi trường hợp khác."""
    cache = []
    a = X
    for layer in layers:
        a_in, extra = a, None
        if isinstance(layer, Dropout):
            if train and layer.p > 0:                                   # inverted dropout
                extra = (rng.random(a_in.shape) >= layer.p).astype(a_in.dtype)
                a = a_in * extra / (1.0 - layer.p)
            else:
                a = a_in
        elif isinstance(layer, Linear):
            a = a_in @ layer.W.T + layer.b                              # z = aWᵀ + b
        else:
            a = layer(a_in)
        cache.append((layer, a_in, a, extra))
    return a, cache


# ---------------------------------------------------------------------------
# Lan truyền ngược
def softmax_cross_entropy(logits, y):
    """Trả về (loss trung bình, delta = dJ/dlogits). log_softmax trừ max trước khi exp.
    BP1 cho softmax + cross-entropy: delta = (Ŷ − Y) / N."""
    N = len(logits)
    log_probs = log_softmax(logits)
    loss = -float(log_probs[np.arange(N), y].mean())
    delta = np.exp(log_probs)
    delta[np.arange(N), y] -= 1.0
    return loss, delta / N


def linear_backward(layer, a_in, delta):
    """BP4, BP3 và nửa đầu BP2: (dW, db, gradient theo đầu vào).
    dW = deltaᵀ a_in, tính bằng (a_inᵀ delta)ᵀ: cùng kết quả nhưng nhanh hơn nhiều trên BLAS của NumPy."""
    dW = (a_in.T @ delta).T
    db = delta.sum(axis=0)
    grad_in = delta @ layer.W
    return dW, db, grad_in


def relu_backward(a_in, grad):
    return grad * (a_in > 0)


def sigmoid_backward(a_out, grad):
    return grad * a_out * (1.0 - a_out)                                 # σ' = σ(1 − σ)


def tanh_backward(a_out, grad):
    return grad * (1.0 - a_out ** 2)                                    # tanh' = 1 − tanh²


def dropout_backward(mask, p, grad):
    return grad if mask is None else grad * mask / (1.0 - p)


def backward(cache, grad_out, weight_decay=0.0):
    """Duyệt cache từ cuối về đầu, ghi dW, db cho mọi lớp Linear; trả về gradient theo đầu vào.
    L2: J + (λ/2) Σ ||W||², nên dJ/dW có thêm λW; không phạt bias."""
    grad = grad_out
    for layer, a_in, a_out, extra in reversed(cache):
        if isinstance(layer, Linear):
            dW, db, grad = linear_backward(layer, a_in, grad)
            layer.dW[...] = dW + weight_decay * layer.W
            layer.db[...] = db
        elif isinstance(layer, Activation):
            if layer.name == "relu":
                grad = relu_backward(a_in, grad)
            elif layer.name == "sigmoid":
                grad = sigmoid_backward(a_out, grad)
            else:
                grad = tanh_backward(a_out, grad)
        elif isinstance(layer, Dropout):
            grad = dropout_backward(extra, layer.p, grad)
        else:
            raise TypeError(f"Lớp không hỗ trợ: {type(layer).__name__}")
    return grad


# ---------------------------------------------------------------------------
# Bộ tối ưu: Δw(t) = α Δw(t−1) − γ ∂J/∂w,  w(t+1) = w(t) + Δw(t)
class SGDMomentum:
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params)
        self.lr, self.momentum = lr, momentum
        self.velocity = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.velocity):
            v *= self.momentum
            v -= self.lr * g
            p += v


def _params_grads(layers):
    ls = linears(layers)
    return [p for l in ls for p in (l.W, l.b)], [g for l in ls for g in (l.dW, l.db)]


# ---------------------------------------------------------------------------
# Vòng lặp huấn luyện
def train_one_epoch(layers, X, y, optimizer, grads, batch_size=128, weight_decay=0.0, rng=None):
    """Một epoch mini-batch: xáo trộn, forward có cache, BP1, backward, cập nhật.
    Trả về loss trung bình trong epoch (để phát hiện phân kỳ)."""
    N = len(X)
    perm = rng.permutation(N)
    total = 0.0
    for s in range(0, N, batch_size):
        idx = perm[s:s + batch_size]
        logits, cache = forward_with_cache(layers, X[idx], train=True, rng=rng)
        loss, delta = softmax_cross_entropy(logits, y[idx])
        backward(cache, delta, weight_decay=weight_decay)
        optimizer.step(grads)
        total += loss * len(idx)
    return total / N


def evaluate_loss_acc(layers, X, y, batch_size=5000):
    """(cross-entropy trung bình, accuracy), dropout tắt."""
    total, correct = 0.0, 0
    for s in range(0, len(X), batch_size):
        logits, _ = forward_with_cache(layers, X[s:s + batch_size], train=False)
        loss, _ = softmax_cross_entropy(logits, y[s:s + batch_size])
        total += loss * len(logits)
        correct += int((logits.argmax(axis=1) == y[s:s + batch_size]).sum())
    return total / len(X), correct / len(X)


# ---------------------------------------------------------------------------
# Bộ phân lớp
class MLPClassifierNP(BaseClassifier):
    """MLP huấn luyện bằng backprop tự viết + SGD momentum.

    Siêu tham số:
        hidden        : tuple số nơ-ron mỗi lớp ẩn, vd (256, 128)
        activation    : "relu" | "sigmoid" | "tanh"
        init          : "he" | "xavier" | "const"
        lr, momentum  : γ và α của SGD momentum
        batch_size    : kích thước mini-batch
        weight_decay  : hệ số L2 λ (không phạt bias)
        p_drop        : tỉ lệ dropout sau mỗi lớp ẩn (0 = không dùng)
        max_epochs    : số epoch tối đa
        patience      : early stopping theo val loss; None = không dùng (cần X_val, y_val)
        seed          : khởi tạo trọng số, thứ tự mini-batch và mặt nạ dropout
    Thuộc tính học được: layers_, history_ (dict các list theo epoch), best_epoch_, n_epochs_, n_params_,
        fit_time_ (cập nhật + val loss cho early stopping, theo định nghĩa chung ở results/README.md),
        diag_time_ (loss/accuracy trên train mỗi epoch, chỉ để vẽ đường học).
    """

    def __init__(self, hidden=(256, 128), activation="relu", init="he", lr=0.05, momentum=0.9,
                 batch_size=128, weight_decay=0.0, p_drop=0.0, max_epochs=30, patience=None, seed=SEED):
        self.hidden = hidden
        self.activation = activation
        self.init = init
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.p_drop = p_drop
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def _sizes(self):
        return [self.n_features_in_, *self.hidden, len(self.classes_)]

    def initial_layers(self):
        """Mạng ở trạng thái khởi tạo (cùng seed), để tách phần học từ dữ liệu khỏi phần khởi tạo."""
        return build_mlp(self._sizes(), self.activation, self.p_drop, self.init, seed=self.seed)

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        X, y = self._validate_fit(X, y)
        if self.patience is not None and X_val is None:
            raise ValueError("Early stopping (patience) cần X_val, y_val.")
        X = X.astype(np.float32)
        has_val = X_val is not None
        if has_val:
            Xv = self._validate_predict(X_val).astype(np.float32)
            yv = np.asarray(y_val, dtype=np.int64)

        self.layers_ = self.initial_layers()
        self.n_params_ = n_params(self.layers_)
        params, grads = _params_grads(self.layers_)
        opt = SGDMomentum(params, self.lr, self.momentum)
        rng = np.random.default_rng(self.seed + 1)                      # thứ tự mini-batch, mặt nạ dropout

        self.history_ = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}
        best = (float("inf"), None, 0)
        wait, t_fit, t_diag = 0, 0.0, 0.0
        for ep in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            # γ quá lớn làm trọng số tràn số (E1 cố ý đo điều này): không in cảnh báo, kiểm tra run_loss bên dưới
            with np.errstate(over="ignore", invalid="ignore"):
                run_loss = train_one_epoch(self.layers_, X, y, opt, grads, self.batch_size, self.weight_decay, rng)
                vl, va = evaluate_loss_acc(self.layers_, Xv, yv) if has_val else (np.nan, np.nan)
            t1 = time.perf_counter()
            tl, ta = evaluate_loss_acc(self.layers_, X, y) if np.isfinite(run_loss) else (np.nan, np.nan)
            t_fit += t1 - t0
            t_diag += time.perf_counter() - t1
            for k, v in zip(self.history_, [tl, ta, vl, va, t_fit]):
                self.history_[k].append(v)
            if verbose:
                print(f"epoch {ep:2d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
            if not np.isfinite(run_loss):                               # phân kỳ
                break
            if self.patience is not None:
                if vl < best[0]:
                    best, wait = (vl, copy.deepcopy(self.layers_), ep), 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        self.n_epochs_ = ep
        self.best_epoch_ = ep
        if self.patience is not None and best[1] is not None:
            self.layers_ = best[1]
            self.best_epoch_ = best[2]
        self.fit_time_ = t_fit
        self.diag_time_ = t_diag
        return self

    # --- dự đoán ---
    def decision_function(self, X):
        X = self._validate_predict(X).astype(np.float32)
        out = [forward_with_cache(self.layers_, X[s:s + 5000], train=False)[0] for s in range(0, len(X), 5000)]
        return np.concatenate(out).astype(np.float64)

    def predict_proba(self, X):
        return np.exp(log_softmax(self.decision_function(X)))

    def predict(self, X):
        idx = self.decision_function(X).argmax(axis=1)                  # kiểm tra đã fit trước khi đọc classes_
        return self.classes_[idx]
