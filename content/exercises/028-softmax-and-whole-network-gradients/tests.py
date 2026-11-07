"""Checks for softmax, cross-entropy, whole-network gradients, and the gradient check."""

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


@check("softmax gives [0.665241, 0.244728, 0.090031] for scores [2, 1, 0], ignores a shift, and survives scores of 1000")
def test_softmax(sub):
    softmax = has(sub, "softmax")
    got = softmax(np.array([[2.0, 1.0, 0.0]]))
    shape(got, (1, 3), label="softmax output")
    close(got, np.array([[0.665241, 0.244728, 0.090031]]), atol=1e-6, label="worked example")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 4)) * 3
    p = softmax(z)
    close(p.sum(axis=1), np.ones(5), atol=1e-12, label="row sums")
    if p.min() <= 0:
        fail("softmax produced a non-positive probability")
    z = np.array([[1.0, -2.0, 0.5]])
    close(softmax(z + 100.0), softmax(z), atol=1e-12, label="softmax(z + 100) against softmax(z)")
    big = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    finite(big, label="softmax at scores 1000, 0, -1000")
    close(big, np.array([[1.0, 0.0, 0.0]]), atol=1e-12, label="softmax at scores 1000, 0, -1000")


@check("log_softmax equals log(softmax) and stays finite at scores of plus or minus 1000")
def test_log_softmax(sub):
    softmax = has(sub, "softmax")
    log_softmax = has(sub, "log_softmax")
    rng = np.random.default_rng(1)
    z = rng.normal(size=(4, 5)) * 2
    close(log_softmax(z), np.log(softmax(z)), atol=1e-10, label="log_softmax against log of softmax")
    big = log_softmax(np.array([[1000.0, 0.0, -1000.0]]))
    finite(big, label="log_softmax at scores 1000, 0, -1000")
    close(big, np.array([[0.0, -1000.0, -2000.0]]), atol=1e-9, label="log_softmax at scores 1000, 0, -1000")


@check("The cross-entropy loss is 0.407606 for scores [2, 1, 0] and class 0, and grows by 1 per class down")
def test_loss_value(sub):
    Tensor = has(sub, "Tensor")
    sce = has(sub, "softmax_cross_entropy")
    z = np.array([[2.0, 1.0, 0.0]])
    for label, want in [(0, 0.407606), (1, 1.407606), (2, 2.407606)]:
        loss = sce(Tensor(z), np.array([label]))
        close(loss.value, want, atol=1e-6, label=f"loss for class {label}")
    shape(loss.value, (), label="loss")


@check("The slope on the logits is (probabilities - one-hot) / n, matches finite differences, and stays finite at scores of 1000")
def test_ce_gradient(sub):
    Tensor = has(sub, "Tensor")
    softmax = has(sub, "softmax")
    sce = has(sub, "softmax_cross_entropy")
    rng = np.random.default_rng(2)
    z = rng.normal(size=(6, 4))
    labels = rng.integers(4, size=6)
    logits = Tensor(z)
    sce(logits, labels).backward()
    want = softmax(z)
    want[np.arange(6), labels] -= 1.0
    want /= 6
    close(logits.grad, want, atol=1e-8, label="against (p - onehot) / n")
    fd = finite_difference(lambda v: float(sce(Tensor(v), labels).value), z)
    close(logits.grad, fd, atol=1e-6, label="against finite differences")

    logits = Tensor(np.array([[2.0, 1.0, 0.0]]))
    sce(logits, np.array([0])).backward()
    close(logits.grad, np.array([[-0.334759, 0.244728, 0.090031]]), atol=1e-6, label="slope on the worked example")

    row = np.array([[2.0, 1.0, 0.0]])
    one = Tensor(row)
    loss1 = sce(one, np.array([0]))
    loss1.backward()
    six = Tensor(np.repeat(row, 6, axis=0))
    loss6 = sce(six, np.zeros(6, dtype=int))
    loss6.backward()
    close(loss6.value, loss1.value, atol=1e-12, label="mean loss of identical rows")
    close(six.grad, np.repeat(one.grad / 6.0, 6, axis=0), atol=1e-12, label="each row's slope in a batch of 6")

    logits = Tensor(np.array([[1000.0, 0.0, 0.0]]))
    loss = sce(logits, np.array([1]))
    loss.backward()
    finite(loss.value, label="loss at scores 1000, 0, 0")
    close(loss.value, 1000.0, atol=1e-6, label="loss when the right class scores 1000 below the top")
    close(logits.grad, np.array([[1.0, -1.0, 0.0]]), atol=1e-9, label="slope at scores 1000, 0, 0")


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


@check("Every weight and bias gradient matches finite differences, and the hand-traced network gives loss 1.701413")
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

    traced = [
        {"W": np.array([[1.0, -1.0], [0.5, 0.5]]), "b": np.array([0.5, -0.5])},
        {"W": np.array([[0.4, -0.2], [1.0, 0.5]]), "b": np.array([0.0, 0.0])},
    ]
    loss, grads = f(traced, np.array([[1.0, 2.0]]), np.array([1]))
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
