"""Checks for the ridge SVD path and degrees-of-freedom exercise."""

import numpy as np

from mlcheck import check, close, fail, shape


def correlated_design(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    X = latent @ rng.normal(size=(3, d)) + 0.2 * rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    return X, y


@check("shrink_factors gives 1/s at lambda 0 and s/(s^2 + lambda) otherwise")
def test_shrink_hand(sub):
    s = np.array([0.1, 1.0])
    got = sub.shrink_factors(s, np.array([0.0, 1.0]))
    shape(got, (2, 2), label="shrink_factors output")
    want = np.array([[10.0, 1.0], [0.1 / 1.01, 0.5]])
    close(got, want, atol=1e-12, label="shrink_factors for s = (0.1, 1), lambda = (0, 1)")


@check("shrink_factors returns 0, not NaN, for a zero singular value at lambda 0")
def test_shrink_zero(sub):
    s = np.array([2.0, 0.0])
    got = sub.shrink_factors(s, np.array([0.0, 1.0]))
    if not np.all(np.isfinite(got)):
        fail(f"shrink_factors returned {got}; a zero singular value at lambda = 0 should give 0")
    close(got, np.array([[0.5, 0.0], [0.4, 0.0]]), atol=1e-12, label="shrink_factors for s = (2, 0)")


@check("ridge_svd_path reproduces the chapter's example: (-1, 3) at lambda 0, (0.908, 0.987) at 1")
def test_path_example(sub):
    X = np.array([[3.0, 3.0], [-0.1, 0.1]])
    y = np.array([6.0, 0.4])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    path = sub.ridge_svd_path(U, s, Vt, y, np.array([0.0, 1.0]))
    shape(path, (2, 2), label="ridge_svd_path output")
    close(path[0], np.array([-1.0, 3.0]), atol=1e-9, label="weights at lambda = 0")
    close(path[1], np.array([0.90815273, 0.98658411]), atol=1e-7, label="weights at lambda = 1")


@check("ridge_svd_path matches ridge_closed_form at every lambda")
def test_path_matches_closed_form(sub):
    X, y = correlated_design(60, 10, seed=3)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.array([0.01, 0.5, 5.0, 50.0, 500.0])
    path = sub.ridge_svd_path(U, s, Vt, y, lambdas)
    shape(path, (5, 10), label="ridge_svd_path output")
    for j, lam in enumerate(lambdas):
        close(path[j], sub.ridge_closed_form(X, y, lam), atol=1e-8, label=f"lambda = {lam:g} against ridge_closed_form")


@check("effective_degrees_of_freedom counts the rank at lambda 0 and falls to 0 for a huge lambda")
def test_dof_ends(sub):
    s = np.array([10.0, 5.0, 1.0, 0.1])
    at_zero = sub.effective_degrees_of_freedom(s, 0.0)
    if abs(at_zero - 4.0) > 1e-9:
        fail(f"at lambda = 0 the count should be the rank 4, got {at_zero:.6g}")
    huge = sub.effective_degrees_of_freedom(s, 1e9)
    if huge > 1e-3:
        fail(f"at a huge lambda the count should be near 0, got {huge:.6g}")
    with_zero = sub.effective_degrees_of_freedom(np.array([2.0, 0.0]), 0.0)
    if not np.isfinite(with_zero) or abs(with_zero - 1.0) > 1e-9:
        fail(f"a zero singular value adds nothing, so s = (2, 0) at lambda 0 should count 1, got {with_zero}")


@check("effective_degrees_of_freedom handles an array of lambdas and falls as lambda grows")
def test_dof_array(sub):
    s = np.array([10.0, 5.0, 1.0, 0.1])
    lambdas = np.array([0.0, 0.01, 1.0, 100.0, 10000.0])
    got = sub.effective_degrees_of_freedom(s, lambdas)
    shape(got, (5,), label="effective_degrees_of_freedom with five lambdas")
    want = np.array([np.sum(s**2 / (s**2 + lam)) for lam in lambdas])
    close(got, want, atol=1e-10, label="against the defining sum")
    if np.any(np.diff(got) >= 0):
        fail(f"the count should fall as lambda grows, got {np.round(got, 4)}")


@check("effective_degrees_of_freedom equals the trace of the ridge smoother matrix")
def test_dof_is_trace(sub):
    X, y = correlated_design(40, 6, seed=8)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    for lam in (0.5, 5.0, 50.0):
        smoother = X @ np.linalg.solve(X.T @ X + lam * np.eye(6), X.T)
        close(
            sub.effective_degrees_of_freedom(s, lam),
            np.trace(smoother),
            atol=1e-8,
            label=f"count against trace of X (X^T X + {lam:g} I)^-1 X^T",
        )
