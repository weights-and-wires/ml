"""Checks for the lost digits exercise."""

import inspect

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def with_condition(kappa: float, n: int = 200, d: int = 8, seed: int = 1):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.logspace(0.0, -np.log10(kappa), d)
    return u @ np.diag(s) @ v.T


NEAR = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]])


@check("condition_number reports sigma_max over sigma_min")
def test_condition_number(sub):
    got = has(sub, "condition_number")
    X = with_condition(1e4, d=5, seed=3)
    almost(got(X), 1e4, tol=1e4 * 1e-6, label="condition number of a built matrix")
    almost(got(np.eye(4)), 1.0, tol=1e-9, label="condition number of the identity")
    houses = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    s = np.linalg.svd(houses, compute_uv=False)
    almost(got(houses), float(s[0] / s[-1]), tol=1e-9, label="condition number of the three houses")
    almost(got(NEAR), 127.94, tol=0.01, label="condition number of the nearly repeated columns")

    rank_deficient = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    value = got(rank_deficient)
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


@check("fit_normal_equations recovers (2/3, 1/2) for the three houses")
def test_houses(sub):
    fit = has(sub, "fit_normal_equations")
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])
    w = fit(X, y)
    shape(w, (2,), label="fit output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights")


@check("fit_normal_equations matches a reference solve on a well-conditioned problem")
def test_well_conditioned(sub):
    fit = has(sub, "fit_normal_equations")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X @ rng.normal(size=5) + 0.1 * rng.normal(size=300)
    want = np.linalg.lstsq(X, y, rcond=None)[0]
    close(fit(X, y), want, rtol=1e-9, atol=1e-10, label="against a reference solve")


@check("A wobble of 0.03 in one price moves the weights from (1, 0) to (0.7, 0.3)")
def test_wobble(sub):
    fit = has(sub, "fit_normal_equations")
    w0 = fit(NEAR, np.array([1.0, 2.0, 3.0]))
    w1 = fit(NEAR, np.array([1.0, 2.0, 3.03]))
    close(w0, np.array([1.0, 0.0]), atol=1e-9, label="weights before the wobble")
    close(w1, np.array([0.7, 0.3]), atol=1e-9, label="weights after the wobble")
    sigma_min = float(np.linalg.svd(NEAR, compute_uv=False)[-1])
    bound = 0.03 / sigma_min
    moved = float(np.linalg.norm(w1 - w0))
    if moved > bound * (1 + 1e-9):
        fail(f"the weights moved by {moved:.3g}, more than the bound wobble / sigma_min = {bound:.3g}")


@check("The solver goes through a Cholesky factor, not lstsq or pinv")
def test_source(sub):
    func = has(sub, "fit_normal_equations")
    try:
        source = inspect.getsource(func)
    except OSError:
        return
    for word in ("lstsq", "pinv"):
        if word in source:
            fail(f"fit_normal_equations calls {word}, which solves the problem for you")
    if "cholesky" not in source:
        fail("fit_normal_equations should factor X^T X with np.linalg.cholesky")


@check("At condition number 1e7 the normal equations are much less accurate than a method that works on X")
def test_lost_digits(sub):
    fit = has(sub, "fit_normal_equations")
    X = with_condition(1e7, n=200, d=8, seed=11)
    rng = np.random.default_rng(12)
    w_true = rng.normal(size=8)
    y = X @ w_true

    def relative_error(w):
        return float(np.abs(w - w_true).max()) / float(np.abs(w_true).max())

    stable = relative_error(np.linalg.lstsq(X, y, rcond=None)[0])
    yours = relative_error(fit(X, y))
    if yours < 100 * stable:
        fail(
            f"your relative error {yours:.3g} is within 100 times of the stable error "
            f"{stable:.3g}, so the solve does not seem to go through X^T X"
        )
    if yours > 1.0:
        fail(f"your relative error is {yours:.3g}; the Cholesky solve should still keep a digit or two")
