"""Checks for the leave-one-out shortcut exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def brute_force_loo(X, y, lam):
    n = X.shape[0]
    errors = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        w = ridge_fit(X[keep], y[keep], lam)
        errors[i] = (y[i] - X[i] @ w) ** 2
    return float(np.mean(errors))


def data(n=50, d=4, seed=1, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + noise * rng.normal(size=n)
    return X, y


HAND_X = np.array([[1.0], [2.0], [3.0]])
HAND_Y = np.array([1.0, 2.0, 2.0])


@check("hat_matrix is n by n, symmetric, and H @ y gives the ridge fitted values")
def test_hat_matrix(sub):
    got = has(sub, "hat_matrix")
    X, y = data()
    for lam in (0.0, 2.0):
        H = got(X, lam)
        shape(H, (50, 50), label=f"hat matrix at lam={lam}")
        close(H, H.T, atol=1e-9, label="H should be symmetric")
        close(H @ y, X @ ridge_fit(X, y, lam), atol=1e-8,
              label=f"H @ y against the ridge fitted values at lam={lam}")


@check("with no penalty and n > d, applying the hat twice changes nothing")
def test_projection(sub):
    got = has(sub, "hat_matrix")
    X, _ = data(n=30, d=3, seed=2)
    H = got(X, 0.0)
    close(H @ H, H, atol=1e-8, label="H @ H against H at lam=0")
    close(np.trace(H), 3.0, atol=1e-8, label="the trace of H equals the number of features")


@check("leverages of the hand example are 1/15, 4/15, and 9/15")
def test_leverages_hand(sub):
    got = has(sub, "leverages")
    close(got(HAND_X, 1.0), np.array([1, 4, 9]) / 15.0, atol=1e-12,
          label="leverages for inputs 1, 2, 3 at lam=1")


@check("leverages lie in [0, 1) and add up to the degrees of freedom, the sum of s^2 / (s^2 + lam)")
def test_leverage_properties(sub):
    got = has(sub, "leverages")
    X, _ = data(n=40, d=6, seed=3)
    s = np.linalg.svd(X, compute_uv=False)
    for lam in (0.1, 1.0, 20.0):
        h = got(X, lam)
        shape(h, (40,), label="leverages")
        if np.any(h < 0) or np.any(h >= 1):
            fail(f"at lam={lam} every leverage should lie in [0, 1), got min {h.min():.4g}, max {h.max():.4g}")
        close(h.sum(), np.sum(s ** 2 / (s ** 2 + lam)), atol=1e-8,
              label=f"sum of leverages at lam={lam}")


@check("loo_linear_smoother returns 0.2869 on the hand example")
def test_loo_hand(sub):
    got = has(sub, "loo_linear_smoother")
    want = (4 / 49 + 64 / 121 + 1 / 4) / 3
    close(got(HAND_X, HAND_Y, 1.0), want, atol=1e-10, label="leave-one-out MSE for inputs 1, 2, 3")


@check("loo_linear_smoother matches n separate refits at three penalties")
def test_loo_matches_refits(sub):
    got = has(sub, "loo_linear_smoother")
    X, y = data(n=50, d=4, seed=1)
    for lam in (0.0, 0.5, 5.0):
        close(got(X, y, lam), brute_force_loo(X, y, lam), rtol=1e-6, atol=1e-9,
              label=f"leave-one-out MSE at lam={lam}")


@check("the honest error is larger than the in-sample error, most of all with a flexible model")
def test_loo_exceeds_training(sub):
    got = has(sub, "loo_linear_smoother")
    X, y = data(n=40, d=20, seed=4, noise=1.0)
    H = has(sub, "hat_matrix")(X, 0.5)
    in_sample = float(np.mean((y - H @ y) ** 2))
    honest = got(X, y, 0.5)
    if honest <= 1.5 * in_sample:
        fail(f"with 20 features for 40 points the leave-one-out error should be far above "
             f"the in-sample error {in_sample:.3f}, got {honest:.3f}")
