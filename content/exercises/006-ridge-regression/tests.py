"""Checks for the ridge regression exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, shape


def random_problem(seed: int, n: int = 25, d: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    return X, y


def correlated_design(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    X = latent @ rng.normal(size=(3, d)) + 0.2 * rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    return X, y


@check("ridge_objective scores the hand example (2.5 at (1.5, 0.5), 5 at (3, 1)) and adds exactly half lambda times the squared length")
def test_objective(sub):
    X = np.eye(2)
    y = np.array([3.0, 1.0])
    almost(sub.ridge_objective(X, y, np.array([1.5, 0.5]), 1.0), 2.5, 1e-12, label="score at (1.5, 0.5)")
    almost(sub.ridge_objective(X, y, np.array([3.0, 1.0]), 1.0), 5.0, 1e-12, label="score at (3, 1)")

    X, y = random_problem(0)
    rng = np.random.default_rng(1)
    w = rng.normal(size=4)
    base = sub.ridge_objective(X, y, w, 0.0)
    almost(base, 0.5 * np.sum((X @ w - y) ** 2), 1e-10, label="score at lam = 0")
    for lam in (0.5, 3.0):
        fee = sub.ridge_objective(X, y, w, lam) - base
        almost(fee, 0.5 * lam * np.dot(w, w), 1e-9, label=f"fee at lam = {lam}")


@check("ridge_gradient matches finite differences of ridge_objective")
def test_gradient_finite_difference(sub):
    X, y = random_problem(2)
    rng = np.random.default_rng(3)
    for lam in (0.0, 0.7, 5.0):
        w = rng.normal(size=4)
        got = sub.ridge_gradient(X, y, w, lam)
        shape(got, (4,), label="ridge_gradient output")
        want = finite_difference(lambda v: sub.ridge_objective(X, y, v, lam), w.copy())
        close(got, want, rtol=1e-5, atol=1e-5, label=f"gradient against finite differences at lam = {lam}")


@check("ridge_closed_form has zero gradient at its answer and agrees with least squares on stacked data")
def test_closed_form(sub):
    X, y = random_problem(4, n=40, d=6)
    for lam in (0.1, 2.0, 50.0):
        w = sub.ridge_closed_form(X, y, lam)
        shape(w, (6,), label="ridge_closed_form output")
        grad = X.T @ (X @ w - y) + lam * w
        close(grad, np.zeros(6), atol=1e-8, label=f"gradient at the solution, lam = {lam}")

    X, y = random_problem(5, n=30, d=5)
    lam = 4.0
    stacked_X = np.vstack([X, np.sqrt(lam) * np.eye(5)])
    stacked_y = np.concatenate([y, np.zeros(5)])
    want = np.linalg.lstsq(stacked_X, stacked_y, rcond=None)[0]
    close(sub.ridge_closed_form(X, y, lam), want, atol=1e-8, label="against the stacked least squares solve")
    close(
        sub.ridge_closed_form(X, y, 0.0),
        np.linalg.lstsq(X, y, rcond=None)[0],
        atol=1e-8,
        label="lam = 0 against plain least squares",
    )


@check("ridge_closed_form splits two identical columns evenly, and the weights shrink as lambda grows")
def test_duplicates_and_shrinkage(sub):
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 2.0])
    w = sub.ridge_closed_form(X, y, 1.0)
    if not np.all(np.isfinite(w)):
        fail(f"ridge_closed_form returned {w} on identical columns; X^T X + lam I is solvable when lam > 0")
    close(w, np.array([5.0 / 11.0, 5.0 / 11.0]), atol=1e-10, label="weights on identical columns at lam = 1")

    X, y = random_problem(6, n=40, d=5)
    norms = [np.linalg.norm(sub.ridge_closed_form(X, y, lam)) for lam in (0.0, 0.5, 5.0, 50.0, 500.0)]
    if not all(a > b for a, b in zip(norms, norms[1:])):
        fail(f"the weight length should fall as lam rises, but the lengths were {np.round(norms, 4)}")
    rng = np.random.default_rng(7)
    lam = 3.0
    w = sub.ridge_closed_form(X, y, lam)
    best = sub.ridge_objective(X, y, w, lam)
    for _ in range(50):
        other = w + 0.05 * rng.normal(size=5)
        if sub.ridge_objective(X, y, other, lam) < best - 1e-12:
            fail("a nearby set of weights scores lower than the closed-form solution, so it is not the bottom")


@check("shrink_factors gives 1/s at lambda 0 and s/(s^2 + lambda) otherwise, and 0 (not NaN) for a zero singular value")
def test_shrink_factors(sub):
    s = np.array([0.1, 1.0])
    got = sub.shrink_factors(s, np.array([0.0, 1.0]))
    shape(got, (2, 2), label="shrink_factors output")
    want = np.array([[10.0, 1.0], [0.1 / 1.01, 0.5]])
    close(got, want, atol=1e-12, label="shrink_factors for s = (0.1, 1), lambda = (0, 1)")

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


@check("effective_degrees_of_freedom counts the rank at lambda 0, falls to 0 for a huge lambda, and handles arrays of lambdas")
def test_dof(sub):
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
