"""Checks for the soft thresholding and lasso exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, shape


@check("soft_threshold maps 3, -3, 0.5, -0.5, 0, 2 to 2, -2, 0, 0, 0, 1 at threshold 1")
def test_soft_threshold_hand(sub):
    x = np.array([3.0, -3.0, 0.5, -0.5, 0.0, 2.0])
    close(sub.soft_threshold(x, 1.0), np.array([2.0, -2.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-12)


@check("soft_threshold keeps the shape and is odd: S(-x) = -S(x)")
def test_soft_threshold_shape_symmetry(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5)) * 2
    out = sub.soft_threshold(x, 0.7)
    shape(out, (4, 5), label="soft_threshold output")
    close(sub.soft_threshold(-x, 0.7), -out, atol=1e-12, label="S(-x) against -S(x)")


@check("soft_threshold returns exactly 0, not a tiny number, for |x| <= threshold")
def test_soft_threshold_exact_zero(sub):
    x = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    out = sub.soft_threshold(x, 1.0)
    equal(np.abs(out), np.zeros(5), label="soft_threshold at threshold 1 for entries inside [-1, 1]")


@check("soft_threshold is the exact minimiser of 0.5 (z - x)^2 + t |z|")
def test_soft_threshold_is_prox(sub):
    rng = np.random.default_rng(4)
    grid = np.linspace(-5.0, 5.0, 4001)
    for x in rng.uniform(-3.0, 3.0, size=8):
        for t in (0.3, 1.0, 2.5):
            z = float(sub.soft_threshold(np.array([x]), t)[0])
            objective = 0.5 * (grid - x) ** 2 + t * np.abs(grid)
            value = 0.5 * (z - x) ** 2 + t * abs(z)
            if value > objective.min() + 1e-6:
                fail(
                    f"soft_threshold({x:.3f}, {t}) = {z:.4f} scores {value:.6f}, but a grid search "
                    f"finds {objective.min():.6f} near z = {grid[np.argmin(objective)]:.4f}"
                )


@check("lasso_objective scores the hand example: 4 at (3, 1), 3.25 at (1.5, 0.5), 3 at (2, 0)")
def test_lasso_objective_hand(sub):
    X = np.eye(2)
    y = np.array([3.0, 1.0])
    for w, want in (([3.0, 1.0], 4.0), ([1.5, 0.5], 3.25), ([2.0, 0.0], 3.0)):
        almost(sub.lasso_objective(X, y, np.array(w), 1.0), want, 1e-12, label=f"score at {tuple(w)}")


@check("lasso_objective is half the squared misses plus lambda times the absolute size, with no half on the fee")
def test_lasso_objective_formula(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    w = rng.normal(size=4)
    base = sub.lasso_objective(X, y, w, 0.0)
    almost(base, 0.5 * np.sum((X @ w - y) ** 2), 1e-10, label="score at lam = 0")
    for lam in (0.5, 3.0):
        almost(sub.lasso_objective(X, y, w, lam) - base, lam * np.sum(np.abs(w)), 1e-9, label=f"fee at lam = {lam}")


@check("lasso_identity solves the chapter's example: (2, 0), not ridge's (1.5, 0.5)")
def test_lasso_identity_example(sub):
    got = sub.lasso_identity(np.array([3.0, 1.0]), 1.0)
    close(got, np.array([2.0, 0.0]), atol=1e-12, label="lasso_identity(y = (3, 1), lam = 1)")
    zeros = sub.lasso_identity(np.array([3.0, -1.0, 0.2]), 3.0)
    equal(zeros, np.zeros(3), label="a fee at least as large as every |y| sets every weight to 0")


@check("no nearby weights beat lasso_identity on the lasso objective")
def test_lasso_identity_is_best(sub):
    rng = np.random.default_rng(5)
    y = rng.normal(size=6) * 2
    lam = 0.8
    w = sub.lasso_identity(y, lam)
    shape(w, (6,), label="lasso_identity output")
    X = np.eye(6)

    def score(v):
        return 0.5 * np.sum((X @ v - y) ** 2) + lam * np.sum(np.abs(v))

    best = score(w)
    for _ in range(300):
        other = w + 0.1 * rng.normal(size=6)
        if score(other) < best - 1e-12:
            fail(f"weights {np.round(other, 3)} score {score(other):.6f}, below lasso_identity's {best:.6f}")
    for i in range(6):
        for delta in (-0.01, 0.01):
            other = w.copy()
            other[i] += delta
            if score(other) < best - 1e-12:
                fail(f"moving weight {i} by {delta} lowers the score, so it is not the exact minimiser")
