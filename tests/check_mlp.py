"""Kiểm tra src/models/mlp.py: lan truyền ngược tự viết, bộ tối ưu và bộ phân lớp.
Chạy: python tests/check_mlp.py   (hoặc python tests/run_all.py)

    - softmax + cross-entropy khớp ví dụ mèo / chó / chim (Bài 4, trang 70, 72)
    - gradient tự viết khớp autograd của PyTorch (mọi loại lớp, có L2 và dropout)
    - SGD momentum đúng công thức trang 65
    - khởi tạo He / Xavier đúng công thức trang 73–74
    - lan truyền thuận khớp sklearn.neural_network.MLPClassifier khi dùng chung trọng số
PyTorch và autograd chỉ được dùng ở đây để đối chiếu.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier as SklearnMLP

from _helpers import Skip, main
from src.models.mlp import (Activation, Dropout, Linear, MLPClassifierNP, SGDMomentum, backward,
                            forward_with_cache, linears, softmax_cross_entropy)


def _blobs(seed=0, n=300, d=20, K=4, noise=0.15):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    centers = rng.uniform(0.0, 1.0, size=(K, d))
    X = np.clip(centers[y] + noise * rng.standard_normal((n, d)), 0.0, 1.0).astype(np.float32)
    return X, y


def check_softmax_cross_entropy_slide_example():
    z = np.array([[2.0, 1.0, 0.1]])
    loss, delta = softmax_cross_entropy(z, np.array([0]))
    assert abs(loss - 0.4170) < 1e-3
    assert np.allclose(delta, [[0.6590 - 1, 0.2424, 0.0986]], atol=1e-3)
    loss2, _ = softmax_cross_entropy(z, np.array([2]))
    assert abs(loss2 - 2.317) < 2e-3
    lo, do = softmax_cross_entropy(np.array([[1000.0, 0.0, -1000.0]]), np.array([0]))
    assert np.isfinite(lo) and np.isfinite(do).all(), "tràn số: phải trừ max trước khi exp"


def check_gradient_matches_autograd():
    try:
        import torch
    except ImportError:
        raise Skip("chưa cài torch")
    rng = np.random.default_rng(0)
    layers = [Linear(20, 16, rng=rng, dtype=np.float64), Activation("relu"), Dropout(0.3),
              Linear(16, 12, init="xavier", rng=rng, dtype=np.float64), Activation("tanh"),
              Linear(12, 8, rng=rng, dtype=np.float64), Activation("sigmoid"),
              Linear(8, 4, rng=rng, dtype=np.float64)]
    X, y = _blobs()
    Xb, yb, lam = X[:64].astype(np.float64), y[:64], 1e-3

    logits, cache = forward_with_cache(layers, Xb, train=True, rng=np.random.default_rng(1))
    loss, delta = softmax_cross_entropy(logits, yb)
    backward(cache, delta, weight_decay=lam)

    acts = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
    params = []
    a = torch.tensor(Xb)
    for layer, _, _, extra in cache:                                    # autograd, cùng mặt nạ dropout
        if isinstance(layer, Linear):
            W = torch.tensor(layer.W, requires_grad=True)
            b = torch.tensor(layer.b, requires_grad=True)
            params.append((layer, W, b))
            a = a @ W.T + b
        elif isinstance(layer, Activation):
            a = acts[layer.name](a)
        else:
            a = a * torch.tensor(extra) / (1 - layer.p)
    ref = torch.nn.functional.cross_entropy(a, torch.tensor(yb))
    total = ref + 0.5 * lam * sum((W ** 2).sum() for _, W, _ in params)
    total.backward()

    assert abs(loss - ref.item()) < 1e-10
    for i, (layer, W, b) in enumerate(params):
        for mine, r in ((layer.dW, W.grad.numpy()), (layer.db, b.grad.numpy())):
            rel = np.linalg.norm(mine - r) / (np.linalg.norm(mine) + np.linalg.norm(r) + 1e-30)
            assert rel < 1e-8, f"gradient sai ở lớp Linear {i}: {rel:.2e}"


def check_sgd_momentum_formula():
    w = np.array([1.0])
    opt = SGDMomentum([w], lr=0.1, momentum=0.9)
    opt.step([np.array([2.0])])                                         # Δ = −0.2, w = 0.8
    assert np.allclose(w, 0.8)
    opt.step([np.array([2.0])])                                         # Δ = 0.9·(−0.2) − 0.2 = −0.38, w = 0.42
    assert np.allclose(w, 0.42)


def check_init_formulas():
    he = Linear(400, 300, init="he", rng=np.random.default_rng(0), dtype=np.float64).W
    assert abs(he.std() - np.sqrt(2 / 400)) < 0.005
    xa = Linear(400, 300, init="xavier", rng=np.random.default_rng(0), dtype=np.float64).W
    assert np.abs(xa).max() <= np.sqrt(6 / 700) and abs(xa.std() - np.sqrt(2 / 700)) < 0.005
    assert np.all(Linear(5, 3, init="const", const=0.02).W == np.float32(0.02))


def check_forward_matches_sklearn():
    """Chép trọng số sang sklearn MLPClassifier rồi so predict_proba."""
    X, y = _blobs()
    ours = MLPClassifierNP(hidden=(16, 8), max_epochs=2, seed=0).fit(X, y)
    sk = SklearnMLP(hidden_layer_sizes=(16, 8), activation="relu", max_iter=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sk.fit(X, y)                                                    # chỉ để sklearn tạo đúng cấu trúc
    lin = linears(ours.layers_)
    sk.coefs_ = [l.W.astype(np.float64).T for l in lin]
    sk.intercepts_ = [l.b.astype(np.float64) for l in lin]
    np.testing.assert_allclose(ours.predict_proba(X), sk.predict_proba(X), atol=1e-5)
    np.testing.assert_array_equal(ours.predict(X), sk.predict(X))


CHECKS = [
    check_softmax_cross_entropy_slide_example,
    check_gradient_matches_autograd,
    check_sgd_momentum_formula,
    check_init_formulas,
    check_forward_matches_sklearn,
]

if __name__ == "__main__":
    main(CHECKS, "mlp")
