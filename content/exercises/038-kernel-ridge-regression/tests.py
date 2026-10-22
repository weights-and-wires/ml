"""Checks for the kernel ridge regression exercise."""

import numpy as np

from mlcheck import check, close, fail, shape


def objective(K, y, lam, alpha):
    return float(np.sum((y - K @ alpha) ** 2) + lam * alpha @ K @ alpha)


@check("kernel_ridge_fit solves (K + lam I) alpha = y")
def test_fit_solves_system(sub):
    rng = np.random.default_rng(2)
    n = 15
    A = rng.normal(size=(n, n))
    K = A @ A.T  # positive semidefinite by construction
    y = rng.normal(size=n)
    lam = 0.4
    alpha = sub.kernel_ridge_fit(K, y, lam)
    shape(alpha, (n,), label="alpha")
    close((K + lam * np.eye(n)) @ alpha, y, atol=1e-8, rtol=1e-6, label="(K + lam I) alpha")


@check("kernel_ridge_fit reproduces the worked two-point example")
def test_fit_hand_example(sub):
    k = np.exp(-0.5)
    K = np.array([[1.0, k], [k, 1.0]])
    y = np.array([1.0, 2.0])
    alpha = sub.kernel_ridge_fit(K, y, 1.0)
    close(alpha, np.array([0.21666095, 0.93429425]), atol=1e-6, label="alpha for lam = 1")
    close(K @ alpha, np.array([0.78333905, 1.06570575]), atol=1e-6, label="training predictions K alpha")


@check("alpha is a stationary point of the objective: its finite-difference gradient is zero")
def test_fit_is_stationary(sub):
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 6, size=8))[:, None]
    K = sub.rbf_kernel(x, x, 1.0)
    y = np.sin(x[:, 0]) + 0.1 * rng.normal(size=8)
    lam = 0.3
    alpha = sub.kernel_ridge_fit(K, y, lam)
    h = 1e-5
    grad = np.zeros(8)
    for i in range(8):
        e = np.zeros(8)
        e[i] = h
        grad[i] = (objective(K, y, lam, alpha + e) - objective(K, y, lam, alpha - e)) / (2 * h)
    close(grad, np.zeros(8), atol=1e-5, label="finite-difference gradient of the objective at alpha")
    best = objective(K, y, lam, alpha)
    for _ in range(5):
        if objective(K, y, lam, alpha + 0.05 * rng.normal(size=8)) < best:
            fail("a random nudge of alpha lowered the objective, so alpha is not the minimiser")


@check("Training predictions equal y - lam alpha; tiny lam interpolates and huge lam shrinks")
def test_lambda_behaviour(sub):
    x = np.array([0.51, 2.37, 3.21, 3.82, 5.93, 6.13])[:, None]
    y = np.array([0.36, 0.73, -0.39, -0.58, -0.30, 0.17])
    K = sub.rbf_kernel(x, x, 1.0)
    lam = 0.1
    alpha = sub.kernel_ridge_fit(K, y, lam)
    close(K @ alpha, y - lam * alpha, atol=1e-8, label="K alpha against y - lam alpha")
    tiny = sub.kernel_ridge_fit(K, y, 1e-8)
    close(K @ tiny, y, atol=1e-5, label="training predictions at lam = 1e-8")
    huge = sub.kernel_ridge_fit(K, y, 1e3)
    if np.linalg.norm(K @ huge) > 0.05 * np.linalg.norm(y):
        fail(f"with lam = 1000 the predictions should shrink toward 0, but their norm is {np.linalg.norm(K @ huge):.3f} against {np.linalg.norm(y):.3f} for y")


@check("kernel_ridge_predict matches a hand-written sum of alpha_i k(x, x_i)")
def test_predict(sub):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 2))
    alpha = rng.normal(size=5)
    queries = rng.normal(size=(3, 2))
    K_test = sub.rbf_kernel(queries, X, 0.8)
    got = sub.kernel_ridge_predict(K_test, alpha)
    shape(got, (3,), label="predictions")
    want = np.array([sum(alpha[i] * np.exp(-0.8 * np.sum((q - X[i]) ** 2)) for i in range(5)) for q in queries])
    close(got, want, atol=1e-10, label="sum of weighted similarities")


@check("A linear kernel reproduces primal ridge regression through w = X^T alpha")
def test_matches_primal_ridge(sub):
    # Hand case: x = (1, 2), y = (1, 2), lam = 1 gives alpha = (1/6, 1/3) and w = 5/6.
    X1 = np.array([[1.0], [2.0]])
    a1 = sub.kernel_ridge_fit(X1 @ X1.T, np.array([1.0, 2.0]), 1.0)
    close(a1, np.array([1 / 6, 1 / 3]), atol=1e-10, label="alpha for the one-dimensional example")
    close(sub.implied_weights(X1, a1), np.array([5 / 6]), atol=1e-10, label="w for the one-dimensional example")

    rng = np.random.default_rng(5)
    n, d = 20, 5
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    lam = 0.7
    alpha = sub.kernel_ridge_fit(X @ X.T, y, lam)
    w = sub.implied_weights(X, alpha)
    shape(w, (d,), label="implied weights")
    w_primal = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    close(w, w_primal, atol=1e-8, rtol=1e-6, label="X^T alpha against primal ridge")
    X_test = rng.normal(size=(8, d))
    close(sub.kernel_ridge_predict(X_test @ X.T, alpha), X_test @ w_primal, atol=1e-8, rtol=1e-6, label="predictions")
