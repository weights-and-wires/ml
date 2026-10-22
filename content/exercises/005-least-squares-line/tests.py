"""Checks for the least squares line exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def houses():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])
    return X, y


def random_problem(seed: int, n: int = 40, d: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    return X, y, rng


@check("loss gives 0.5 for the line y = x and 1/12 for the best line")
def test_loss_houses(sub):
    loss = has(sub, "loss")
    X, y = houses()
    almost(loss(X, y, np.array([0.0, 1.0])), 0.5, tol=1e-12, label="loss at (0, 1)")
    almost(loss(X, y, np.array([2 / 3, 1 / 2])), 1 / 12, tol=1e-12, label="loss at (2/3, 1/2)")


@check("loss is half the sum of squared misses on random data")
def test_loss_random(sub):
    loss = has(sub, "loss")
    X, y, rng = random_problem(1)
    for _ in range(3):
        w = rng.normal(size=X.shape[1])
        want = 0.5 * float(np.sum((X @ w - y) ** 2))
        almost(loss(X, y, w), want, tol=1e-9 * max(1.0, want), label="loss")


@check("gradient equals X^T (X w - y) on the three houses")
def test_gradient_houses(sub):
    gradient = has(sub, "gradient")
    X, y = houses()
    got = gradient(X, y, np.array([0.0, 1.0]))
    shape(got, (2,), label="gradient output")
    close(got, np.array([1.0, 3.0]), atol=1e-12, label="gradient at (0, 1)")


@check("gradient matches finite differences of loss")
def test_gradient_finite_difference(sub):
    gradient = has(sub, "gradient")
    # The reference loss lives here, so this check does not depend on your loss.
    X, y, rng = random_problem(2)
    ref = lambda w: 0.5 * float(np.sum((X @ w - y) ** 2))
    for _ in range(3):
        w = rng.normal(size=X.shape[1])
        got = gradient(X, y, w)
        shape(got, w.shape, label="gradient output")
        eps = 1e-6
        want = np.zeros_like(w)
        for j in range(len(w)):
            step = np.zeros_like(w)
            step[j] = eps
            want[j] = (ref(w + step) - ref(w - step)) / (2 * eps)
        close(got, want, rtol=1e-5, atol=1e-5, label="gradient against finite differences")


@check("fit_normal_equations recovers (2/3, 1/2) for the three houses")
def test_fit_houses(sub):
    fit = has(sub, "fit_normal_equations")
    X, y = houses()
    w = fit(X, y)
    shape(w, (2,), label="fit output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights")


@check("The gradient is zero at the fitted weights, and no random nudge lowers the loss")
def test_bottom(sub):
    fit = has(sub, "fit_normal_equations")
    X, y, rng = random_problem(3)
    w = fit(X, y)
    g = X.T @ (X @ w - y)
    scale = float(np.abs(X.T @ y).max())
    if np.abs(g).max() > 1e-9 * scale:
        fail(f"the gradient at your weights is {g}, which should be about zero")
    base = 0.5 * float(np.sum((X @ w - y) ** 2))
    for _ in range(20):
        nudged = w + 0.01 * rng.normal(size=w.shape)
        if 0.5 * float(np.sum((X @ nudged - y) ** 2)) < base - 1e-12:
            fail("a small random nudge lowered the loss, so your weights are not the bottom")
