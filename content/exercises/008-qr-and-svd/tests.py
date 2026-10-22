"""Checks for the QR and SVD exercise."""

import inspect

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

HOUSES = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
PRICES = np.array([1.0, 2.0, 2.0])
REPEATED = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])


def well_conditioned(n: int = 300, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0, -0.25])[:d]
    y = X @ w_true + 0.05 * rng.normal(size=n)
    return X, y, w_true


def with_condition(kappa: float, n: int = 200, d: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.logspace(0.0, -np.log10(kappa), d)
    return u @ np.diag(s) @ v.T


@check("fit_qr recovers (2/3, 1/2) for the three houses")
def test_qr_houses(sub):
    fit = has(sub, "fit_qr")
    w = fit(HOUSES, PRICES)
    shape(w, (2,), label="fit_qr output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights")


@check("fit_qr matches a reference solve and leaves a residual orthogonal to the columns")
def test_qr_random(sub):
    fit = has(sub, "fit_qr")
    X, y, _ = well_conditioned()
    w = fit(X, y)
    shape(w, (X.shape[1],), label="fit_qr output")
    close(w, np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-9, atol=1e-10, label="against a reference solve")
    scale = float(np.abs(X.T @ y).max())
    if np.abs(X.T @ (y - X @ w)).max() > 1e-9 * scale:
        fail("the residual of fit_qr is not orthogonal to the columns of X")


@check("fit_svd recovers (2/3, 1/2) for the three houses and matches a reference solve")
def test_svd_full_rank(sub):
    fit = has(sub, "fit_svd")
    w = fit(HOUSES, PRICES)
    shape(w, (2,), label="fit_svd output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights for the houses")
    X, y, _ = well_conditioned(seed=4)
    close(fit(X, y), np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-9, atol=1e-10, label="against a reference solve")


@check("fit_svd returns the shortest weights (3/35, 6/35) for the repeated columns")
def test_svd_repeated(sub):
    fit = has(sub, "fit_svd")
    w = fit(REPEATED, np.ones(3))
    shape(w, (2,), label="fit_svd output")
    if float(np.abs(w).max()) > 1e3:
        fail(f"the weights are huge ({w}); drop the near-zero singular value instead of dividing by it")
    close(w, np.array([3 / 35, 6 / 35]), atol=1e-9, label="minimum-norm weights")


@check("fit_svd has no component in the null space on a rank-deficient table")
def test_minimum_norm(sub):
    fit = has(sub, "fit_svd")
    rng = np.random.default_rng(7)
    base = rng.normal(size=(60, 3))
    X = np.column_stack([base, base @ np.array([1.0, -2.0, 0.5]), base.sum(axis=1)])
    y = base @ np.array([1.0, 2.0, -1.0]) + 0.01 * rng.normal(size=60)
    w = fit(X, y, 1e-10)
    shape(w, (5,), label="fit_svd output")
    gap = float(np.abs(X @ w - X @ np.linalg.lstsq(X, y, rcond=None)[0]).max())
    if gap > 1e-8:
        fail(f"the fitted values differ from the least squares fit by up to {gap:.3g}")
    null_space = np.linalg.svd(X)[2][3:]
    leakage = float(np.abs(null_space @ w).max())
    if leakage > 1e-8:
        fail(
            f"the shortest weights have no part in the null space, but yours has {leakage:.3g}. "
            "Drop the singular values at or below rcond * s[0]."
        )


@check("rank counts the singular values above the cut-off")
def test_rank(sub):
    got = has(sub, "rank")
    equal_to = lambda value, want, label: (
        None if int(value) == want else fail(f"{label}: expected {want}, got {value}")
    )
    equal_to(got(np.eye(4)), 4, "rank of the identity")
    equal_to(got(HOUSES), 2, "rank of the three houses")
    equal_to(got(REPEATED), 1, "rank of the repeated columns")
    equal_to(got(np.zeros((3, 2))), 0, "rank of the zero matrix")
    rng = np.random.default_rng(5)
    base = rng.normal(size=(40, 3))
    wide = np.column_stack([base, base[:, 0] + base[:, 1], base[:, 2] * 2.0])
    equal_to(got(wide), 3, "rank of a table with two dependent columns")


@check("At condition number 1e7, QR and the SVD keep about nine digits")
def test_stable(sub):
    X = with_condition(1e7, n=200, d=8, seed=11)
    rng = np.random.default_rng(12)
    w_true = rng.normal(size=8)
    y = X @ w_true
    for name in ("fit_qr", "fit_svd"):
        w = has(sub, name)(X, y)
        error = float(np.abs(w - w_true).max()) / float(np.abs(w_true).max())
        if error > 1e-6:
            fail(
                f"{name} has relative error {error:.3g}, but a method that works on X "
                "should keep about nine digits here. Check that it never forms X^T X."
            )


@check("The solvers are built from factorisations, not from lstsq, pinv, or X^T X")
def test_no_shortcut(sub):
    for name in ("fit_qr", "fit_svd"):
        func = has(sub, name)
        try:
            source = inspect.getsource(func)
        except OSError:
            continue
        if "NotImplementedError" in source:
            fail(f"{name} is not written yet")
        for word in ("lstsq", "pinv", "X.T @ X"):
            if word in source:
                fail(f"{name} uses {word}. Build it from np.linalg.qr or np.linalg.svd instead.")
