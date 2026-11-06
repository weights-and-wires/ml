"""Checks for the whole-network gradients and the gradient check."""

import copy

import numpy as np

from mlcheck import check, close, fail, finite, finite_difference, has, shape


def make_net(seed=0, n=6, d=3, hidden=4, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(classes, size=n)
    params = [
        {"W": rng.normal(size=(d, hidden)) * 0.5, "b": rng.normal(size=hidden) * 0.1},
        {"W": rng.normal(size=(hidden, classes)) * 0.5, "b": rng.normal(size=classes) * 0.1},
    ]
    return params, X, y


def plain_loss(params, X, y):
    h = X
    for i, layer in enumerate(params):
        h = h @ layer["W"] + layer["b"]
        if i < len(params) - 1:
            h = np.maximum(h, 0.0)
    h = h - h.max(axis=1, keepdims=True)
    log_probs = h - np.log(np.exp(h).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(y)), y].mean())


@check("mlp_loss_and_grad returns a float loss equal to a plain numpy forward pass, and matching shapes")
def test_forward(sub):
    f = has(sub, "mlp_loss_and_grad")
    params, X, y = make_net()
    loss, grads = f(params, X, y)
    if not isinstance(loss, float):
        fail(f"loss should be a Python float, got {type(loss).__name__}")
    close(loss, plain_loss(params, X, y), atol=1e-10, label="loss against a plain forward pass")
    if len(grads) != len(params):
        fail(f"expected {len(params)} gradient dicts, got {len(grads)}")
    for i, (g, p) in enumerate(zip(grads, params)):
        shape(g["W"], p["W"].shape, label=f"layer {i} W gradient")
        shape(g["b"], p["b"].shape, label=f"layer {i} b gradient")


@check("The first layer's weights and the last layer's bias match finite differences")
def test_gradients(sub):
    f = has(sub, "mlp_loss_and_grad")
    params, X, y = make_net(seed=5)
    _, grads = f(params, X, y)

    def loss_of(i, key, value):
        trial = copy.deepcopy(params)
        trial[i][key] = value
        return plain_loss(trial, X, y)

    close(grads[0]["W"], finite_difference(lambda v: loss_of(0, "W", v), params[0]["W"]), atol=1e-5, label="W of the first layer")
    close(grads[1]["b"], finite_difference(lambda v: loss_of(1, "b", v), params[1]["b"]), atol=1e-5, label="b of the last layer")
    close(grads[0]["b"], finite_difference(lambda v: loss_of(0, "b", v), params[0]["b"]), atol=1e-5, label="b of the first layer")
    close(grads[1]["W"], finite_difference(lambda v: loss_of(1, "W", v), params[1]["W"]), atol=1e-5, label="W of the last layer")


@check("The hand-traced network gives loss 1.701413 and the slopes from the chapter")
def test_trace(sub):
    f = has(sub, "mlp_loss_and_grad")
    params = [
        {"W": np.array([[1.0, -1.0], [0.5, 0.5]]), "b": np.array([0.5, -0.5])},
        {"W": np.array([[0.4, -0.2], [1.0, 0.5]]), "b": np.array([0.0, 0.0])},
    ]
    loss, grads = f(params, np.array([[1.0, 2.0]]), np.array([1]))
    close(loss, 1.701413, atol=1e-6, label="loss")
    close(grads[1]["W"], np.array([[2.043936, -2.043936], [0.0, 0.0]]), atol=1e-6, label="last-layer W slope")
    close(grads[1]["b"], np.array([0.817574, -0.817574]), atol=1e-6, label="last-layer b slope")
    close(grads[0]["W"], np.array([[0.490545, 0.0], [0.981089, 0.0]]), atol=1e-6, label="first-layer W slope")
    close(grads[0]["b"], np.array([0.490545, 0.0]), atol=1e-6, label="first-layer b slope")


@check("numeric_grad matches a known gradient, and leaves the caller's array alone")
def test_numeric_grad(sub):
    numeric_grad = has(sub, "numeric_grad")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 2))
    original = x.copy()
    got = numeric_grad(lambda a: float((a**3).sum() + 2.0 * a[0, 1]), x)
    want = 3.0 * x**2
    want[0, 1] += 2.0
    shape(got, x.shape, label="numeric gradient")
    close(got, want, atol=1e-6, label="gradient of sum(x^3) + 2 x[0, 1]")
    close(x, original, atol=0.0, label="the array you passed in")


@check("gradient_check reports an error below 1e-6 for a correct network")
def test_gradient_check(sub):
    gradient_check = has(sub, "gradient_check")
    params, X, y = make_net(seed=7)
    err = gradient_check(params, X, y)
    finite(err, label="gradient_check result")
    if not (0.0 <= err < 1e-6):
        fail(f"gradient_check returned {err:.3e}; a correct network should give less than 1e-6")


@check("gradient_check catches a wrong gradient: scaling every gradient by 2 gives an error near 1/3")
def test_gradient_check_catches(sub):
    gradient_check = has(sub, "gradient_check")
    true_f = sub.mlp_loss_and_grad
    params, X, y = make_net(seed=8)

    def broken(p, x, labels):
        loss, grads = true_f(p, x, labels)
        return loss, [{"W": 2.0 * g["W"], "b": 2.0 * g["b"]} for g in grads]

    sub.mlp_loss_and_grad = broken
    try:
        err = gradient_check(params, X, y)
    finally:
        sub.mlp_loss_and_grad = true_f
    if err < 0.1:
        fail(f"gradient_check returned {err:.3e} for gradients that are twice too large; expected about 0.333")
    close(err, 1.0 / 3.0, atol=1e-3, label="worst relative error with doubled gradients")


@check("One sgd_step lowers the loss, leaves the old parameters alone, and 40 steps cut it a lot")
def test_sgd(sub):
    f = has(sub, "mlp_loss_and_grad")
    sgd_step = has(sub, "sgd_step")
    params, X, y = make_net(seed=9, n=20, d=4, hidden=8)
    snapshot = copy.deepcopy(params)
    loss0, grads = f(params, X, y)
    new = sgd_step(params, grads, 0.1)
    for a, b in zip(params, snapshot):
        close(a["W"], b["W"], atol=0.0, label="old W")
        close(a["b"], b["b"], atol=0.0, label="old b")
    loss1, _ = f(new, X, y)
    if not loss1 < loss0:
        fail(f"one step moved the loss from {loss0:.4f} to {loss1:.4f}; it should fall")
    p = new
    for _ in range(40):
        _, g = f(p, X, y)
        p = sgd_step(p, g, 0.3)
    loss_end, _ = f(p, X, y)
    if not loss_end < 0.7 * loss0:
        fail(f"after 41 steps the loss is {loss_end:.4f} from {loss0:.4f}; expected below {0.7 * loss0:.4f}")
