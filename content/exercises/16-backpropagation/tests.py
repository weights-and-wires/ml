"""Checks for the reverse-mode autodiff exercise."""

import copy

import numpy as np

from mlcheck import check, close, fail, finite_difference, has


@check("Tensor stores its value and starts with a zero gradient of the same shape")
def test_tensor_init(sub):
    Tensor = has(sub, "Tensor")
    value = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = Tensor(value)
    close(t.value, value, label="Tensor.value")
    close(t.grad, np.zeros_like(value), label="Tensor.grad before backward")


@check("__add__'s gradient matches finite differences, including a broadcast bias")
def test_add(sub):
    Tensor = has(sub, "Tensor")
    rng = np.random.default_rng(0)
    a_val = rng.normal(size=(5, 3))
    b_val = rng.normal(size=(3,))  # broadcasts over the first axis, like a bias

    def total(bv):
        a = Tensor(a_val)
        b = Tensor(bv)
        return float((a + b).sum().value)

    fd = finite_difference(total, b_val)
    a, b = Tensor(a_val), Tensor(b_val)
    (a + b).sum().backward()
    close(b.grad, fd, atol=1e-5, label="gradient with respect to the broadcast operand")
    close(a.grad, np.ones_like(a_val), atol=1e-10, label="gradient with respect to the other operand")


@check("__mul__'s gradient matches finite differences")
def test_mul(sub):
    Tensor = has(sub, "Tensor")
    rng = np.random.default_rng(1)
    a_val = rng.normal(size=(4, 3))
    b_val = rng.normal(size=(4, 3))

    def total(av):
        a = Tensor(av)
        b = Tensor(b_val)
        return float((a * b).sum().value)

    fd = finite_difference(total, a_val)
    a, b = Tensor(a_val), Tensor(b_val)
    (a * b).sum().backward()
    close(a.grad, fd, atol=1e-5, label="gradient with respect to a")
    close(b.grad, a_val, atol=1e-8, label="gradient with respect to b is a, by the product rule")


@check("__matmul__'s gradient matches finite differences for both operands")
def test_matmul(sub):
    Tensor = has(sub, "Tensor")
    rng = np.random.default_rng(2)
    A_val = rng.normal(size=(4, 5))
    B_val = rng.normal(size=(5, 3))

    def total_A(Av):
        return float((Tensor(Av) @ Tensor(B_val)).sum().value)

    def total_B(Bv):
        return float((Tensor(A_val) @ Tensor(Bv)).sum().value)

    fd_A = finite_difference(total_A, A_val)
    fd_B = finite_difference(total_B, B_val)
    A, B = Tensor(A_val), Tensor(B_val)
    (A @ B).sum().backward()
    close(A.grad, fd_A, atol=1e-5, label="gradient with respect to A")
    close(B.grad, fd_B, atol=1e-5, label="gradient with respect to B")


@check("relu's gradient matches finite differences away from the kink")
def test_relu(sub):
    Tensor = has(sub, "Tensor")
    rng = np.random.default_rng(3)
    x_val = rng.normal(size=8) * 2.0
    x_val = np.where(np.abs(x_val) < 0.05, 0.3, x_val)

    def total(xv):
        return float(Tensor(xv).relu().sum().value)

    fd = finite_difference(total, x_val)
    x = Tensor(x_val)
    x.relu().sum().backward()
    close(x.grad, fd, atol=1e-5, label="relu gradient")
    close(x.grad, (x_val > 0).astype(float), atol=1e-10, label="relu gradient is the indicator of positivity")


@check("Gradients accumulate rather than overwrite when a tensor is used twice in one op")
def test_accumulation(sub):
    Tensor = has(sub, "Tensor")
    a = Tensor(3.0)
    y = a * a
    y.backward()
    # d(a^2)/da = 2a = 6. An implementation that assigns rather than
    # accumulates the two contributions from __mul__'s two operands (both a)
    # would leave a.grad at 3.0 instead.
    close(a.grad, np.array(6.0), atol=1e-10, label="gradient of a*a at a=3")


@check("backward's topological sort handles a diamond-shaped graph correctly")
def test_diamond(sub):
    Tensor = has(sub, "Tensor")
    a = Tensor(2.0)
    b = a * a  # b = a^2
    c = a + a  # c = 2a
    d = b + c  # d = a^2 + 2a, dd/da = 2a + 2
    d.backward()
    close(a.grad, np.array(2 * 2.0 + 2.0), atol=1e-10, label="gradient through both paths of the diamond")


@check("softmax_cross_entropy's gradient matches finite differences")
def test_softmax_cross_entropy(sub):
    Tensor = has(sub, "Tensor")
    sce = has(sub, "softmax_cross_entropy")
    rng = np.random.default_rng(4)
    logits_val = rng.normal(size=(6, 4))
    labels = rng.integers(4, size=6)

    def total(lv):
        return float(sce(Tensor(lv), labels).value)

    fd = finite_difference(total, logits_val)
    logits = Tensor(logits_val)
    loss = sce(logits, labels)
    loss.backward()
    close(logits.grad, fd, atol=1e-5, label="softmax-cross-entropy gradient")

    probs = np.exp(logits_val - logits_val.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    want = probs.copy()
    want[np.arange(6), labels] -= 1.0
    want /= 6
    close(logits.grad, want, atol=1e-8, label="against the closed-form softmax(z) - one_hot(y), over n")


@check("mlp_loss_and_grad matches finite differences end to end on a small network")
def test_mlp_loss_and_grad(sub):
    mlp_loss_and_grad = has(sub, "mlp_loss_and_grad")
    rng = np.random.default_rng(5)
    n, d, hidden, classes = 6, 3, 4, 3
    X = rng.normal(size=(n, d))
    y = rng.integers(classes, size=n)
    params = [
        {"W": rng.normal(size=(d, hidden)) * 0.5, "b": rng.normal(size=hidden) * 0.1},
        {"W": rng.normal(size=(hidden, classes)) * 0.5, "b": rng.normal(size=classes) * 0.1},
    ]

    loss, grads = mlp_loss_and_grad(params, X, y)
    if len(grads) != 2:
        fail(f"expected one gradient dict per layer, got {len(grads)}")

    def loss_of(layer_index, key, value):
        trial = copy.deepcopy(params)
        trial[layer_index][key] = value
        l, _ = mlp_loss_and_grad(trial, X, y)
        return l

    fd_W0 = finite_difference(lambda W: loss_of(0, "W", W), params[0]["W"])
    fd_b1 = finite_difference(lambda b: loss_of(1, "b", b), params[1]["b"])
    close(grads[0]["W"], fd_W0, atol=1e-4, label="gradient of the first layer's weights")
    close(grads[1]["b"], fd_b1, atol=1e-4, label="gradient of the second layer's bias")
