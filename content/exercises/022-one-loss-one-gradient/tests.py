"""Checks for the one-loss-one-gradient exercise."""

import numpy as np

from mlcheck import check, close, fail, shape

X_SHOP = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
Y_SHOP = np.array([1.0, 1.0, 2.0])


def softplus_ref(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def dataset(family, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    w_true = np.array([0.2, 0.5, -0.4])
    eta = X @ w_true
    if family == "gaussian":
        y = eta + rng.normal(size=n)
    elif family == "bernoulli":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return X, y


@check("glm_loss is 3 for the shop counts at zero weights")
def test_loss_hand(sub):
    got = sub.glm_loss("poisson", np.zeros(2), X_SHOP, Y_SHOP)
    close(np.array(got), np.array(3.0), atol=1e-12, label="Poisson loss at zero weights")
    w = np.array([0.05, 0.1])
    eta = X_SHOP @ w
    close(np.array(sub.glm_loss("poisson", w, X_SHOP, Y_SHOP)), np.array(np.sum(np.exp(eta) - Y_SHOP * eta)),
          atol=1e-12, label="Poisson loss at w = (0.05, 0.1)")


@check("glm_loss is half the squared error plus a constant for the Gaussian, and the log loss for the Bernoulli")
def test_loss_cases(sub):
    X, y = dataset("gaussian")
    w = np.array([0.3, -0.2, 0.1])
    want = 0.5 * np.sum((X @ w - y) ** 2) - 0.5 * np.sum(y ** 2)
    close(np.array(sub.glm_loss("gaussian", w, X, y)), np.array(want), atol=1e-9, label="Gaussian loss")
    X, y = dataset("bernoulli")
    z = X @ w
    close(np.array(sub.glm_loss("bernoulli", w, X, y)), np.array(np.sum(softplus_ref(z) - y * z)),
          atol=1e-9, label="Bernoulli loss")


@check("glm_gradient is [-1, -2] for the shop counts at zero weights")
def test_gradient_hand(sub):
    g = sub.glm_gradient("poisson", np.zeros(2), X_SHOP, Y_SHOP)
    shape(g, (2,), label="glm_gradient output")
    close(g, np.array([-1.0, -2.0]), atol=1e-12, label="shop gradient")


@check("glm_gradient matches finite differences of glm_loss for every family")
def test_gradient_fd(sub):
    for family in ("gaussian", "bernoulli", "poisson"):
        X, y = dataset(family)
        w = np.array([0.1, 0.2, -0.1])
        eps = 1e-6
        numeric = np.zeros(3)
        for j in range(3):
            step = np.zeros(3)
            step[j] = eps
            numeric[j] = (sub.glm_loss(family, w + step, X, y) - sub.glm_loss(family, w - step, X, y)) / (2 * eps)
        got = sub.glm_gradient(family, w, X, y)
        shape(got, (3,), label=f"{family} gradient")
        close(got, numeric, rtol=1e-5, atol=1e-4, label=f"{family} gradient")


@check("gradient_descent_glm lowers the loss at every step and keeps steps + 1 entries")
def test_descent_monotone(sub):
    w, history = sub.gradient_descent_glm("poisson", X_SHOP, Y_SHOP, lr=0.05, steps=300)
    shape(w, (2,), label="gradient_descent_glm weights")
    if len(history) != 301:
        fail(f"history has {len(history)} entries, expected 301 (the loss at zero weights plus one per step)")
    close(np.array(history[0]), np.array(3.0), atol=1e-12, label="history[0]")
    if any(b > a + 1e-12 for a, b in zip(history, history[1:])):
        fail("the loss went up on some step; check the sign of the update")


@check("gradient descent reaches the bottom of the shop bowl")
def test_descent_shop(sub):
    w, history = sub.gradient_descent_glm("poisson", X_SHOP, Y_SHOP, lr=0.05, steps=500)
    close(w, np.array([-0.14507, 0.38416]), atol=1e-3, label="weights after 500 steps")
    close(np.array(history[-1]), np.array(2.65951), atol=1e-4, label="loss after 500 steps")
    g = sub.glm_gradient("poisson", w, X_SHOP, Y_SHOP)
    if np.linalg.norm(g) > 1e-2:
        fail(f"gradient norm at the final weights is {np.linalg.norm(g):.3g}, expected near 0")


@check("on Gaussian data, gradient descent approaches least squares")
def test_descent_gaussian(sub):
    X, y = dataset("gaussian", n=50)
    lr = 1.0 / np.linalg.eigvalsh(X.T @ X).max()
    w, _ = sub.gradient_descent_glm("gaussian", X, y, lr=lr, steps=1500)
    want = np.linalg.lstsq(X, y, rcond=None)[0]
    close(w, want, atol=1e-4, label="Gaussian weights vs least squares")
