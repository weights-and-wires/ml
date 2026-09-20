"""Checks for the least squares exercise."""

import inspect

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def well_conditioned(n: int = 400, d: int = 6, seed: int = 0):
    """A design matrix with a condition number near one, and clean targets."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0, -0.25])[:d]
    y = X @ w_true + 0.05 * rng.normal(size=n)
    return X, y, w_true


def with_condition(kappa: float, n: int = 200, d: int = 8, seed: int = 1):
    """A design matrix built to have exactly the requested condition number."""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.logspace(0.0, -np.log10(kappa), d)
    return u @ np.diag(s) @ v.T


@check("condition_number reports sigma_max over sigma_min")
def test_condition_number(sub):
    got = has(sub, "condition_number")
    X = with_condition(1e4, d=5, seed=3)
    almost(got(X), 1e4, tol=1e4 * 1e-6, label="condition number of a built matrix")

    identity = np.eye(4)
    almost(got(identity), 1.0, tol=1e-9, label="condition number of the identity")

    rank_deficient = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    value = got(rank_deficient)
    # The second column is twice the first, so sigma_min is zero in exact
    # arithmetic and rounding noise in floating point. Either answer is right.
    if value < 1e12:
        fail(
            "a rank-deficient matrix has sigma_min at or near zero, so the "
            f"condition number is enormous, but you returned {value:.3g}"
        )


@check("residual_orthogonality is the infinity norm of X^T (y - X w)")
def test_residual_orthogonality(sub):
    got = has(sub, "residual_orthogonality")
    X, y, _ = well_conditioned(n=50, d=3)
    w = np.array([1.0, 2.0, 3.0])
    want = float(np.abs(X.T @ (y - X @ w)).max())
    almost(got(X, y, w), want, tol=1e-8 * max(1.0, want), label="away from the optimum")

    # At the exact solution the residual is orthogonal to the column space, so
    # this drops to rounding error.
    exact = np.linalg.lstsq(X, y, rcond=None)[0]
    value = got(X, y, exact)
    if value > 1e-8:
        fail(
            "at the least squares solution the residual must be orthogonal to "
            f"every column of X, so this should be near zero, but it is {value:.3g}"
        )


@check("fit_qr solves a well-conditioned problem")
def test_fit_qr(sub):
    got = has(sub, "fit_qr")
    X, y, w_true = well_conditioned()
    w = got(X, y)
    shape(w, (X.shape[1],), label="fit_qr output")
    close(w, w_true, atol=0.02, label="coefficients")
    reference = np.linalg.lstsq(X, y, rcond=None)[0]
    close(w, reference, rtol=1e-9, atol=1e-10, label="against a reference solve")


@check("fit_normal_equations solves a well-conditioned problem")
def test_fit_normal_equations(sub):
    got = has(sub, "fit_normal_equations")
    X, y, w_true = well_conditioned()
    w = got(X, y)
    shape(w, (X.shape[1],), label="fit_normal_equations output")
    reference = np.linalg.lstsq(X, y, rcond=None)[0]
    close(w, reference, rtol=1e-8, atol=1e-9, label="against a reference solve")


@check("fit_svd solves a well-conditioned problem")
def test_fit_svd(sub):
    got = has(sub, "fit_svd")
    X, y, _ = well_conditioned()
    w = got(X, y)
    shape(w, (X.shape[1],), label="fit_svd output")
    reference = np.linalg.lstsq(X, y, rcond=None)[0]
    close(w, reference, rtol=1e-9, atol=1e-10, label="against a reference solve")


@check("All three routes put the residual orthogonal to the column space")
def test_all_three_project(sub):
    X, y, _ = well_conditioned()
    scale = float(np.abs(X.T @ y).max())
    for name in ("fit_normal_equations", "fit_qr", "fit_svd"):
        w = has(sub, name)(X, y)
        value = float(np.abs(X.T @ (y - X @ w)).max())
        if value > 1e-8 * scale:
            fail(
                f"{name} leaves a residual that is not orthogonal to the columns: "
                f"||X^T r||_inf is {value:.3g}, against a scale of {scale:.3g}"
            )


@check("fit_svd returns the minimum-norm solution on a rank-deficient design")
def test_minimum_norm(sub):
    got = has(sub, "fit_svd")
    rng = np.random.default_rng(7)
    base = rng.normal(size=(60, 3))
    # The fourth and fifth columns are combinations of the first three, so the
    # null space has dimension two and the solution set is a plane.
    X = np.column_stack([base, base @ np.array([1.0, -2.0, 0.5]), base.sum(axis=1)])
    y = base @ np.array([1.0, 2.0, -1.0]) + 0.01 * rng.normal(size=60)

    w = got(X, y, 1e-10)
    shape(w, (5,), label="fit_svd output on a rank-deficient design")

    fitted_gap = float(np.abs(X @ w - X @ np.linalg.lstsq(X, y, rcond=None)[0]).max())
    if fitted_gap > 1e-8:
        fail(
            "your solution does not reach the least squares fit on a rank-deficient "
            f"design: the fitted values differ by up to {fitted_gap:.3g}"
        )

    null_space = np.linalg.svd(X)[2][3:]
    leakage = float(np.abs(null_space @ w).max())
    if leakage > 1e-8:
        fail(
            "the minimum-norm solution has no component in the null space of X, "
            f"but yours projects onto it with magnitude {leakage:.3g}. Drop the "
            "singular values at or below rcond * s.max() instead of inverting them."
        )


@check("The normal equations lose the digits the condition number predicts")
def test_conditioning_gap(sub):
    X = with_condition(1e7, n=200, d=8, seed=11)
    rng = np.random.default_rng(12)
    w_true = rng.normal(size=8)
    y = X @ w_true

    errors = {}
    for name in ("fit_normal_equations", "fit_qr", "fit_svd"):
        w = has(sub, name)(X, y)
        errors[name] = float(np.abs(w - w_true).max()) / float(np.abs(w_true).max())

    stable = max(errors["fit_qr"], errors["fit_svd"])
    if stable > 1e-6:
        fail(
            "QR and the SVD should keep about nine digits at this conditioning, "
            f"but their relative error is {stable:.3g}. Check that neither route "
            "forms X^T X."
        )
    if errors["fit_normal_equations"] < stable * 100:
        fail(
            "fit_normal_equations is as accurate as the stable routes, so it is "
            "not going through X^T X. Its relative error is "
            f"{errors['fit_normal_equations']:.3g} against {stable:.3g} for QR."
        )


@check("The solvers are built from factorisations, not from lstsq or pinv")
def test_no_shortcut(sub):
    banned = ("lstsq", "pinv")
    for name in ("fit_normal_equations", "fit_qr", "fit_svd"):
        func = has(sub, name)
        try:
            source = inspect.getsource(func)
        except OSError:
            continue
        for word in banned:
            if word in source:
                fail(
                    f"{name} calls {word}, which solves the problem for you. Build "
                    "it from cholesky, qr, or svd instead."
                )
