"""Checks for the least squares and projection exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def houses():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])
    return X, y


def random_problem(seed: int, n: int = 40, d: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    return X, y, rng


@check("loss gives 0.5 for the line y = x and 1/12 for the best line, and is half the sum of squared misses on random data")
def test_loss(sub):
    loss = has(sub, "loss")
    X, y = houses()
    almost(loss(X, y, np.array([0.0, 1.0])), 0.5, tol=1e-12, label="loss at (0, 1)")
    almost(loss(X, y, np.array([2 / 3, 1 / 2])), 1 / 12, tol=1e-12, label="loss at (2/3, 1/2)")
    X, y, rng = random_problem(1)
    for _ in range(3):
        w = rng.normal(size=X.shape[1])
        want = 0.5 * float(np.sum((X @ w - y) ** 2))
        almost(loss(X, y, w), want, tol=1e-9 * max(1.0, want), label="loss on random data")


@check("gradient equals X^T (X w - y) on the three houses and matches finite differences of the loss")
def test_gradient(sub):
    gradient = has(sub, "gradient")
    X, y = houses()
    got = gradient(X, y, np.array([0.0, 1.0]))
    shape(got, (2,), label="gradient output")
    close(got, np.array([1.0, 3.0]), atol=1e-12, label="gradient at (0, 1)")
    # The reference loss lives here, so this check does not depend on your loss.
    X, y, rng = random_problem(2)
    ref = lambda w: 0.5 * float(np.sum((X @ w - y) ** 2))
    for _ in range(3):
        w = rng.normal(size=X.shape[1])
        got = gradient(X, y, w)
        shape(got, w.shape, label="gradient output")
        eps = 1e-6
        want = np.zeros_like(w)
        for j in range(len(w)):
            step = np.zeros_like(w)
            step[j] = eps
            want[j] = (ref(w + step) - ref(w - step)) / (2 * eps)
        close(got, want, rtol=1e-5, atol=1e-5, label="gradient against finite differences")


@check("fit_normal_equations recovers (2/3, 1/2) for the houses, and the gradient is zero at its weights")
def test_fit(sub):
    fit = has(sub, "fit_normal_equations")
    X, y = houses()
    w = fit(X, y)
    shape(w, (2,), label="fit output")
    close(w, np.array([2 / 3, 1 / 2]), atol=1e-12, label="weights")
    X, y, rng = random_problem(3)
    w = fit(X, y)
    g = X.T @ (X @ w - y)
    scale = float(np.abs(X.T @ y).max())
    if np.abs(g).max() > 1e-9 * scale:
        fail(f"the gradient at your weights is {g}, which should be about zero")
    base = 0.5 * float(np.sum((X @ w - y) ** 2))
    for _ in range(20):
        nudged = w + 0.01 * rng.normal(size=w.shape)
        if 0.5 * float(np.sum((X @ nudged - y) ** 2)) < base - 1e-12:
            fail("a small random nudge lowered the loss, so your weights are not the bottom")


@check("residual_orthogonality gives 3, 5, and 0 for the three candidate lines, and matches X^T (y - X w) on random data")
def test_orthogonality(sub):
    got = has(sub, "residual_orthogonality")
    X, y = houses()
    almost(got(X, y, np.array([0.0, 1.0])), 3.0, tol=1e-12, label="weights (0, 1)")
    almost(got(X, y, np.array([1.0, 0.0])), 5.0, tol=1e-12, label="weights (1, 0)")
    almost(got(X, y, np.array([2 / 3, 1 / 2])), 0.0, tol=1e-12, label="weights (2/3, 1/2)")
    X, y, rng = random_problem(4, n=60)
    for _ in range(3):
        w = rng.normal(size=X.shape[1])
        want = float(np.abs(X.T @ (y - X @ w)).max())
        almost(got(X, y, w), want, tol=1e-9 * max(1.0, want), label="check value on random data")


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
    X, y, _ = random_problem(5, n=60)
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
    X, y, rng = random_problem(6, n=60)
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
    X, y, rng = random_problem(7, n=60)
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
