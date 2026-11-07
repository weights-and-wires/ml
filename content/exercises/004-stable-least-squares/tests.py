"""Checks for the stable least squares exercise."""

import inspect

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

NEAR = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]])
HOUSES = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
PRICES = np.array([1.0, 2.0, 2.0])
REPEATED = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])


def with_condition(kappa: float, n: int = 200, d: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.logspace(0.0, -np.log10(kappa), d)
    return u @ np.diag(s) @ v.T


def well_conditioned(n: int = 300, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0, -0.25])[:d]
    y = X @ w_true + 0.05 * rng.normal(size=n)
    return X, y


def relative_error(w, w_true):
    return float(np.abs(w - w_true).max()) / float(np.abs(w_true).max())


@check("condition_number reports sigma_max over sigma_min")
def test_condition_number(sub):
    got = has(sub, "condition_number")
    X = with_condition(1e4, d=5, seed=3)
    almost(got(X), 1e4, tol=1e4 * 1e-6, label="condition number of a built matrix")
    almost(got(np.eye(4)), 1.0, tol=1e-9, label="condition number of the identity")
    s = np.linalg.svd(HOUSES, compute_uv=False)
    almost(got(HOUSES), float(s[0] / s[-1]), tol=1e-9, label="condition number of the three houses")
    almost(got(NEAR), 127.94, tol=0.01, label="condition number of the nearly repeated columns")

    value = got(REPEATED)
    # sigma_min is zero in exact arithmetic and rounding noise in floating point.
    # Either infinity or an enormous finite number is a correct answer.
    if value < 1e12:
        fail(
            "the second column is twice the first, so sigma_min is at or near zero and "
            f"the condition number is enormous, but you returned {value:.3g}"
        )


@check("digits_kept follows 16 - k and 16 - 2k, and never goes below zero")
def test_digits_kept(sub):
    got = has(sub, "digits_kept")
    almost(got(1e7, False), 9.0, tol=1e-9, label="digits with X at kappa 1e7")
    almost(got(1e7, True), 2.0, tol=1e-9, label="digits through X^T X at kappa 1e7")
    almost(got(1.0, True), 16.0, tol=1e-9, label="digits at kappa 1")
    almost(got(1e3, True), 10.0, tol=1e-9, label="digits through X^T X at kappa 1e3")
    almost(got(1e9, True), 0.0, tol=1e-12, label="digits through X^T X at kappa 1e9 (clamped)")


@check("fit_normal_equations solves the houses and shows the wobble from (1, 0) to (0.7, 0.3)")
def test_normal_equations(sub):
    fit = has(sub, "fit_normal_equations")
    w = fit(HOUSES, PRICES)
    shape(w, (2,), label="fit output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights for the three houses")
    X, y = well_conditioned()
    close(fit(X, y), np.linalg.lstsq(X, y, rcond=None)[0], rtol=1e-9, atol=1e-10, label="against a reference solve")

    w0 = fit(NEAR, np.array([1.0, 2.0, 3.0]))
    w1 = fit(NEAR, np.array([1.0, 2.0, 3.03]))
    close(w0, np.array([1.0, 0.0]), atol=1e-9, label="weights before the wobble")
    close(w1, np.array([0.7, 0.3]), atol=1e-9, label="weights after the wobble")
    sigma_min = float(np.linalg.svd(NEAR, compute_uv=False)[-1])
    bound = 0.03 / sigma_min
    moved = float(np.linalg.norm(w1 - w0))
    if moved > bound * (1 + 1e-9):
        fail(f"the weights moved by {moved:.3g}, more than the bound wobble / sigma_min = {bound:.3g}")


@check("fit_qr recovers (2/3, 1/2), matches a reference solve, and leaves an orthogonal residual")
def test_qr(sub):
    fit = has(sub, "fit_qr")
    w = fit(HOUSES, PRICES)
    shape(w, (2,), label="fit_qr output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights for the three houses")
    X, y = well_conditioned()
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
    X, y = well_conditioned(seed=4)
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


@check("At condition number 1e7, the normal equations lose digits that QR and the SVD keep")
def test_lost_digits(sub):
    X = with_condition(1e7, n=200, d=8, seed=11)
    rng = np.random.default_rng(12)
    w_true = rng.normal(size=8)
    y = X @ w_true

    stable = relative_error(np.linalg.lstsq(X, y, rcond=None)[0], w_true)
    yours = relative_error(has(sub, "fit_normal_equations")(X, y), w_true)
    if yours < 100 * stable:
        fail(
            f"your relative error {yours:.3g} is within 100 times of the stable error "
            f"{stable:.3g}, so the solve does not seem to go through X^T X"
        )
    if yours > 1.0:
        fail(f"your relative error is {yours:.3g}; the Cholesky solve should still keep a digit or two")

    for name in ("fit_qr", "fit_svd"):
        error = relative_error(has(sub, name)(X, y), w_true)
        if error > 1e-6:
            fail(
                f"{name} has relative error {error:.3g}, but a method that works on X "
                "should keep about nine digits here. Check that it never forms X^T X."
            )


@check("Each solver is built from its factorisation, not from lstsq or pinv")
def test_source(sub):
    for name, needed in (
        ("fit_normal_equations", "cholesky"),
        ("fit_qr", "qr"),
        ("fit_svd", "svd"),
    ):
        func = has(sub, name)
        try:
            source = inspect.getsource(func)
        except OSError:
            continue
        if "NotImplementedError" in source:
            fail(f"{name} is not written yet")
        words = ["lstsq", "pinv"] + (["X.T @ X"] if name != "fit_normal_equations" else [])
        for word in words:
            if word in source:
                fail(f"{name} uses {word}. Build it from np.linalg.{needed} instead.")
        if needed not in source:
            fail(f"{name} should call np.linalg.{needed}")
