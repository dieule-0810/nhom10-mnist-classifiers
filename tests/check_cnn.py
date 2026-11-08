"""Kiểm tra CNN tự cài (src/models/cnn.py). PyTorch chỉ dùng để đối chiếu, thiếu torch thì bỏ qua phần đó.
Chạy: python tests/check_cnn.py
"""

import numpy as np

from _helpers import Skip, main
from src.models._numeric import log_softmax
from src.models.cnn import Adam, Conv2D, Dense, MaxPool2x2, build_layers


def _torch():
    try:
        import torch
    except ImportError:
        raise Skip("chưa cài torch")
    return torch


def _numeric_grad(f, a, n=12, eps=1e-6, seed=0):
    """Sai phân trung tâm tại n vị trí ngẫu nhiên của mảng a (sửa tại chỗ rồi trả lại)."""
    rng = np.random.default_rng(seed)
    idx = [tuple(rng.integers(0, s) for s in a.shape) for _ in range(n)]
    out = []
    for i in idx:
        old = a[i]
        a[i] = old + eps
        fp = f()
        a[i] = old - eps
        fm = f()
        a[i] = old
        out.append((fp - fm) / (2 * eps))
    return idx, np.array(out)


def _check_layer_grads(layer, x, rel=1e-5):
    """Hàm mất mát giả L = Σ out·R; so gradient giải tích theo x và theo mọi tham số với sai phân."""
    R = np.random.default_rng(1).standard_normal(layer.forward(x).shape)
    f = lambda: float((layer.forward(x) * R).sum())
    layer.forward(x)
    dx = layer.backward(R)
    for arr, grad in [(x, dx)] + list(zip(layer.params, layer.grads)):
        grad = grad.copy()
        idx, num = _numeric_grad(f, arr)
        ana = np.array([grad[i] for i in idx])
        assert np.allclose(ana, num, rtol=rel, atol=1e-7), f"gradient lệch: {ana} vs {num}"


def check_dense_gradient():
    rng = np.random.default_rng(0)
    _check_layer_grads(Dense(7, 5, rng=rng, dtype=np.float64), rng.standard_normal((4, 7)))


def check_network_gradient():
    """Cả mạng nhỏ (ảnh 8×8, 2 khối conv) với cross-entropy: gradient theo tham số và theo ảnh."""
    rng = np.random.default_rng(0)
    layers = build_layers((2, 3), hidden=5, rng=rng, dtype=np.float64, image_shape=(8, 8))
    x = rng.standard_normal((4, 8, 8, 1))
    y = np.array([0, 3, 9, 3])

    def loss():
        z = x
        for layer in layers:
            z = layer.forward(z)
        return -float(log_softmax(z)[np.arange(len(y)), y].mean())

    loss()
    z = x
    for layer in layers:
        z = layer.forward(z)
    d = np.exp(log_softmax(z))
    d[np.arange(len(y)), y] -= 1
    d /= len(y)
    for layer in reversed(layers[1:]):
        d = layer.backward(d)
    dx = layers[0].backward(d, need_dx=True)
    pairs = [(x, dx)] + [(p, g.copy()) for layer in layers for p, g in zip(layer.params, layer.grads)]
    for arr, grad in pairs:
        idx, num = _numeric_grad(loss, arr, n=6)
        ana = np.array([grad[i] for i in idx])
        assert np.allclose(ana, num, rtol=1e-4, atol=1e-8), f"gradient mạng lệch: {ana} vs {num}"


def check_conv_matches_torch():
    torch = _torch()
    F = torch.nn.functional
    rng = np.random.default_rng(0)
    conv = Conv2D(3, 4, rng=rng, dtype=np.float64)
    x = rng.standard_normal((2, 7, 5, 3))
    R = rng.standard_normal((2, 7, 5, 4))
    out = conv.forward(x)
    dx = conv.backward(R)

    xt = torch.tensor(x.transpose(0, 3, 1, 2), requires_grad=True)          # NHWC -> NCHW
    Wt = torch.tensor(conv.W, requires_grad=True)
    bt = torch.tensor(conv.b, requires_grad=True)
    ot = F.conv2d(xt, Wt, bt, padding=1)
    (ot * torch.tensor(R.transpose(0, 3, 1, 2))).sum().backward()
    assert np.allclose(out, ot.detach().numpy().transpose(0, 2, 3, 1))
    assert np.allclose(dx, xt.grad.numpy().transpose(0, 2, 3, 1))
    assert np.allclose(conv.grads[0], Wt.grad.numpy())
    assert np.allclose(conv.grads[1], bt.grad.numpy())


def check_maxpool_matches_torch():
    torch = _torch()
    F = torch.nn.functional
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 6, 8, 3))
    R = rng.standard_normal((2, 3, 4, 3))
    pool = MaxPool2x2()
    out = pool.forward(x)
    dx = pool.backward(R)
    xt = torch.tensor(x.transpose(0, 3, 1, 2), requires_grad=True)
    ot = F.max_pool2d(xt, 2)
    (ot * torch.tensor(R.transpose(0, 3, 1, 2))).sum().backward()
    assert np.allclose(out, ot.detach().numpy().transpose(0, 2, 3, 1))
    assert np.allclose(dx, xt.grad.numpy().transpose(0, 2, 3, 1))


def check_adam_matches_torch():
    torch = _torch()
    rng = np.random.default_rng(0)
    p = rng.standard_normal(5)
    pt = torch.tensor(p.copy(), requires_grad=True)
    opt, opt_t = Adam([p], lr=0.01), torch.optim.Adam([pt], lr=0.01)
    for _ in range(5):
        g = rng.standard_normal(5)
        opt.step([g])
        pt.grad = torch.tensor(g)
        opt_t.step()
    assert np.allclose(p, pt.detach().numpy())


CHECKS = [
    check_dense_gradient,
    check_network_gradient,
    check_conv_matches_torch,
    check_maxpool_matches_torch,
    check_adam_matches_torch,
]

if __name__ == "__main__":
    main(CHECKS, "CNN")
