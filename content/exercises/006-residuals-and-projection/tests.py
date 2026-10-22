"""Checks for the residuals and projection exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def houses():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])
    return X, y


def random_problem(seed: int, n: int = 60, d: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.5 * rng.normal(size=n)
    return X, y, rng


@check("residual_orthogonality gives 3, 5, and 0 for the three candidate lines")
def test_houses(sub):
    got = has(sub, "residual_orthogonality")
    X, y = houses()
    almost(got(X, y, np.array([0.0, 1.0])), 3.0, tol=1e-12, label="weights (0, 1)")
    almost(got(X, y, np.array([1.0, 0.0])), 5.0, tol=1e-12, label="weights (1, 0)")
    almost(got(X, y, np.array([2 / 3, 1 / 2])), 0.0, tol=1e-12, label="weights (2/3, 1/2)")


@check("residual_orthogonality matches the largest entry of X^T (y - X w)")
def test_random_orthogonality(sub):
    got = has(sub, "residual_orthogonality")
    X, y, rng = random_problem(0)
    for _ in range(3):
        w = rng.normal(size=X.shape[1])
        want = float(np.abs(X.T @ (y - X @ w)).max())
        almost(got(X, y, w), want, tol=1e-9 * max(1.0, want), label="check value")


@check("project returns the fitted values 7/6, 5/3, 13/6 for the houses")
def test_project_houses(sub):
    project = has(sub, "project")
    X, y = houses()
    out = project(X, y)
    shape(out, (3,), label="project output")
    close(out, np.array([7 / 6, 5 / 3, 13 / 6]), atol=1e-12, label="fitted values")


@check("The projection leaves a residual orthogonal to every column, and Pythagoras holds")
def test_projection_properties(sub):
    project = has(sub, "project")
    X, y, _ = random_problem(1)
    p = project(X, y)
    shape(p, y.shape, label="project output")
    r = y - p
    scale = float(np.abs(X.T @ y).max())
    if np.abs(X.T @ r).max() > 1e-9 * scale:
        fail(f"the residual is not orthogonal to the columns: X^T r has largest entry {np.abs(X.T @ r).max():.3g}")
    almost(float(y @ y), float(p @ p + r @ r), tol=1e-8 * float(y @ y), label="||y||^2 = ||p||^2 + ||r||^2")


@check("No other point of the column space is closer to y than the projection")
def test_closest(sub):
    project = has(sub, "project")
    X, y, rng = random_problem(2)
    p = project(X, y)
    best = float(np.sum((y - p) ** 2))
    for _ in range(30):
        other = X @ rng.normal(size=X.shape[1])
        if float(np.sum((y - other) ** 2)) < best - 1e-9:
            fail("found a point of the column space that is closer to y than your projection")
    # Projecting twice changes nothing, because p is already on the sheet.
    close(project(X, p), p, atol=1e-9, label="projecting a point already on the sheet")


@check("is_best_fit accepts the best weights, rejects nudged ones, and ignores the scale of y")
def test_is_best_fit(sub):
    best = has(sub, "is_best_fit")
    X, y, rng = random_problem(3)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    if not best(X, y, w, 1e-8):
        fail("the least squares weights were rejected; check the tolerance and the scale")
    if best(X, y, w + 1e-3 * rng.normal(size=w.shape), 1e-8):
        fail("weights nudged by about 1e-3 were accepted")
    big = 1e6
    if not best(X, big * y, big * w, 1e-8):
        fail("the answer changed when y and w were both multiplied by a million; use a relative tolerance")
    Xh, yh = houses()
    if best(Xh, yh, np.array([0.0, 1.0]), 1e-8):
        fail("the weights (0, 1) for the houses were accepted")
