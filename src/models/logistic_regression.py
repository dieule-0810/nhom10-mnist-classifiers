"""Logistic Regression đa lớp (softmax regression), tự cài bằng NumPy, huấn luyện bằng gradient descent.

Logistic Regression nhị phân mở rộng sang K lớp, theo đúng lựa chọn
ở mục 5.2 của EDA: một mô hình softmax thay vì 10 mô hình one-vs-rest, để có xác suất P(y | x)
đúng nghĩa (cần cho log-loss và mức tự tin ở giả thuyết H1).

Ký hiệu:
    x̃ = [x; 1] ∈ R^{d+1}, bias là HÀNG CUỐI của W ∈ R^{(d+1)×K}
    z = W^T x̃,   p = softmax(z),   Y là one-hot của nhãn

Hàm mất mát cross-entropy có điều chuẩn L2 (không phạt bias) [Bishop, mục 4.3.4]:

    J(W) = (1/N) Σ_i [ log Σ_k exp(z_ik) − z_{i,y_i} ] + (λ/2) ‖W_{1:d}‖²_F

    ∇J = (1/N) X̃^T (P − Y) + λ M ⊙ W,     M = [1, …, 1, 0]^T (hàng bias bằng 0)

Với K = 2, softmax suy biến về sigmoid của Logistic Regression nhị phân (thêm một bậc tự do thừa).

Bộ tối ưu: gradient descent, có thêm mini-batch và momentum:
    Δw(t) = α Δw(t−1) − η ∇J,   w(t+1) = w(t) + Δw(t),   Δw(0) = 0
    batch_size=None, momentum=0 -> gradient descent full-batch thuần.

Theo dõi loss theo epoch (EDA 5.2: "theo dõi loss theo vòng lặp để chọn learning rate và số
vòng"); tuỳ chọn early stopping trên validation.

Thời gian (results/README.md, định nghĩa chung): fit_time_ chỉ gồm các bước cập nhật; val_time_ là
thời gian tính val loss (cần cho early stopping, tính vào fit_time của mô hình); diag_time_ là thời
gian tính loss/accuracy trên train mỗi epoch, chỉ để vẽ đường học (KHÔNG tính vào fit_time).
"""

import time

import numpy as np

from src.models._numeric import add_bias, log_softmax, one_hot, softmax
from src.models.base import BaseClassifier


class SoftmaxRegressionNP(BaseClassifier):
    """Logistic Regression đa lớp (softmax), gradient descent, điều chuẩn L2 (không phạt bias).

    Siêu tham số:
        eta          : learning rate η
        lam          : hệ số L2 λ
        max_epochs   : số epoch tối đa (một epoch = một lượt qua toàn tập train)
        batch_size   : None = full-batch; số nguyên = mini-batch có xáo trộn
        momentum     : α trong công thức momentum; 0 = gradient descent thường
        tol          : dừng khi |J(t) − J(t−1)| < tol; None = không dùng
        patience     : số epoch liên tiếp val loss không giảm thì dừng và khôi phục trọng số
                       tốt nhất; None = không early stopping
        eval_every   : ghi lịch sử mỗi eval_every epoch (và ở epoch cuối) để đỡ tốn thời gian
    """

    # Mặc định = gradient descent full-batch thuần (η = 0,5, λ = 1e-3, 2000 epoch, dừng khi |ΔJ| < 1e-7).
    # Mini-batch, momentum, early stopping chỉ bật khi truyền tham số tương ứng.
    def __init__(self, eta=0.5, lam=1e-3, max_epochs=2000, batch_size=None, momentum=0.0,
                 tol=1e-7, patience=None, seed=0, eval_every=1, dtype=np.float32, verbose=False):
        self.eta = eta
        self.lam = lam
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.tol = tol
        self.patience = patience
        self.seed = seed
        self.eval_every = eval_every
        self.dtype = dtype
        self.verbose = verbose

    # ------------------------------------------------------------------ mất mát và gradient
    def _mask(self, d):
        m = np.ones((d, 1), dtype=self.dtype)
        m[-1] = 0.0                                          # không điều chuẩn bias
        return m

    def loss(self, Xb, y, W=None):
        """J(W) trên (X̃, y): cross-entropy trung bình + (λ/2)‖W_{1:d}‖²."""
        W = self.W_ if W is None else W
        logp = log_softmax(Xb @ W)
        ce = -logp[np.arange(len(y)), y].mean()
        return float(ce + 0.5 * self.lam * np.sum(W[:-1] ** 2))

    def gradient(self, Xb, Y, W=None):
        """∇J = X̃^T (P − Y) / N + λ M ⊙ W."""
        W = self.W_ if W is None else W
        P = softmax(Xb @ W)
        return Xb.T @ (P - Y) / len(Xb) + self.lam * self._mask(W.shape[0]) * W

    # ------------------------------------------------------------------ huấn luyện
    def fit(self, X, y, X_val=None, y_val=None):
        Xb = add_bias(np.asarray(X, dtype=self.dtype))
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError("Nhãn phải là 0..K-1.")
        N, d = Xb.shape
        K = len(self.classes_)
        Y = one_hot(y, K).astype(self.dtype)
        has_val = X_val is not None
        if has_val:
            Xb_val = add_bias(np.asarray(X_val, dtype=self.dtype))
            y_val = np.asarray(y_val, dtype=np.int64)
        if self.patience is not None and not has_val:
            raise ValueError("Early stopping (patience) cần X_val, y_val.")

        rng = np.random.default_rng(self.seed)
        self.W_ = np.zeros((d, K), dtype=self.dtype)        # J lồi nên khởi tạo 0 là đủ
        V = np.zeros_like(self.W_)                           # Δw(0) = 0
        bs = N if self.batch_size is None else min(self.batch_size, N)
        full_batch = bs == N

        self.loss_ = []                                      # J(W) trên train, sau mỗi lần ghi
        self.history_ = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time"]}
        best = (np.inf, None, 0)                             # (val loss, W, epoch)
        wait, t_total = 0, 0.0
        self.diag_time_ = self.val_time_ = 0.0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            order = np.arange(N) if full_batch else rng.permutation(N)
            for start in range(0, N, bs):
                idx = order[start:start + bs]
                grad = self.gradient(Xb[idx], Y[idx])
                V = self.momentum * V - self.eta * grad      # Δw = α Δw − η ∇J
                self.W_ += V                                 # w = w + Δw
            t_total += time.perf_counter() - t0

            if epoch % self.eval_every and epoch != self.max_epochs:
                continue
            t_diag = time.perf_counter()
            J = self.loss(Xb, y)
            self.loss_.append(J)
            h = self.history_
            h["epoch"].append(epoch)
            h["train_loss"].append(J)
            h["train_acc"].append(float((self._scores(Xb).argmax(1) == y).mean()))
            h["time"].append(t_total)
            t_val = time.perf_counter()
            self.diag_time_ += t_val - t_diag
            if has_val:
                vl = self.loss(Xb_val, y_val) - 0.5 * self.lam * np.sum(self.W_[:-1] ** 2)
                h["val_loss"].append(vl)                     # val loss = cross-entropy thuần
                t_acc = time.perf_counter()
                self.val_time_ += t_acc - t_val
                h["val_acc"].append(float((self._scores(Xb_val).argmax(1) == y_val).mean()))
                self.diag_time_ += time.perf_counter() - t_acc
            if self.verbose:
                msg = f"epoch {epoch:4d} | {t_total:6.1f}s | J {J:.4f} acc {h['train_acc'][-1]:.4f}"
                if has_val:
                    msg += f" | val loss {h['val_loss'][-1]:.4f} acc {h['val_acc'][-1]:.4f}"
                print(msg)

            if not np.isfinite(J):
                if self.verbose:
                    print("Phân kỳ (loss = nan/inf), dừng sớm.")
                break
            if self.patience is not None:
                if h["val_loss"][-1] < best[0]:
                    best, wait = (h["val_loss"][-1], self.W_.copy(), epoch), 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if self.tol is not None and len(self.loss_) > 1 and abs(self.loss_[-2] - J) < self.tol:
                break

        self.n_epochs_ = epoch
        self.best_epoch_ = epoch
        if self.patience is not None and best[1] is not None:
            self.W_, self.best_epoch_ = best[1], best[2]
        self.fit_time_ = t_total
        return self

    # ------------------------------------------------------------------ dự đoán
    def _scores(self, Xb):
        return Xb @ self.W_

    def decision_function(self, X):
        """z = W^T x̃ cho mọi lớp, shape (N, K)."""
        return self._scores(add_bias(np.asarray(X, dtype=self.dtype)))

    def predict_log_proba(self, X):
        return log_softmax(self.decision_function(X).astype(np.float64))

    def predict_proba(self, X):
        """P(y = k | x), shape (N, K)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    @property
    def weights_(self):
        """Trọng số không gồm bias, shape (d, K) - dùng để vẽ ảnh trọng số."""
        return self.W_[:-1]

    def __repr__(self):
        return (f"SoftmaxRegressionNP(eta={self.eta}, lam={self.lam}, batch_size={self.batch_size}, "
                f"momentum={self.momentum}, max_epochs={self.max_epochs})")