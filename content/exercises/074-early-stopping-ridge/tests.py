"""Checks for the early-stopping exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


@check("gd_quadratic_1d matches (1 - (1 - step * eigenvalue) ** k) * w_star, even when a step overshoots")
def test_gd_loop(sub):
    gd_quadratic_1d = has(sub, "gd_quadratic_1d")
    for eigenvalue, step, w_star in ((2.0, 0.1, 1.0), (0.5, 0.05, -3.0), (1.0, 1.5, 2.0)):
        for k in (0, 1, 2, 10, 25):
            want = (1.0 - (1.0 - step * eigenvalue) ** k) * w_star
            got = gd_quadratic_1d(eigenvalue, step, w_star, k)
            almost(got, want, tol=1e-9, label=f"weight at c={eigenvalue}, step={step}, k={k}")
    almost(gd_quadratic_1d(2.0, 0.1, 1.0, 10), 0.89263, tol=1e-5, label="worked example at k = 10")


@check("early_stopping_equivalent_ridge is infinite at 0 steps, 8 at 1 step, and 0.24058 at 10 steps")
def test_penalty_table(sub):
    ridge = has(sub, "early_stopping_equivalent_ridge")
    got0 = ridge(2.0, 0.1, 0)
    if not np.isinf(got0):
        fail(f"expected float('inf') at 0 steps, got {got0}")
    almost(ridge(2.0, 0.1, 1), 8.0, tol=1e-9, label="penalty at k = 1")
    almost(ridge(2.0, 0.1, 10), 0.24058, tol=1e-5, label="penalty at k = 10")


@check("k steps of gradient descent land on the ridge solution that the penalty predicts")
def test_gd_matches_ridge(sub):
    gd_quadratic_1d = has(sub, "gd_quadratic_1d")
    ridge = has(sub, "early_stopping_equivalent_ridge")
    for eigenvalue, step, w_star in ((2.0, 0.1, 1.0), (0.1, 0.05, 4.0), (5.0, 0.1, -1.5), (20.0, 0.05, 3.0)):
        for k in (1, 5, 25):
            lam = ridge(eigenvalue, step, k)
            ridge_w = eigenvalue / (eigenvalue + lam) * w_star
            got = gd_quadratic_1d(eigenvalue, step, w_star, k)
            almost(ridge_w, got, tol=1e-8, label=f"ridge solution vs gradient descent at c={eigenvalue}, k={k}")


@check("the penalty falls as steps grow, and is smaller on steep directions")
def test_penalty_shape(sub):
    ridge = has(sub, "early_stopping_equivalent_ridge")
    values = [ridge(2.0, 0.1, k) for k in range(1, 40)]
    if any(b >= a for a, b in zip(values, values[1:])):
        fail("the penalty should fall at every extra step, but it did not")
    almost(ridge(2.0, 0.1, 5000), 0.0, tol=1e-9, label="penalty after many steps")
    flat, middle, steep = (ridge(c, 0.1, 25) for c in (0.1, 1.0, 5.0))
    if not (flat > middle > steep):
        fail(f"at 25 steps expected penalties to fall as curvature rises, got {flat:.4g}, {middle:.4g}, {steep:.4g}")


@check("early_stopping_equivalent_ridge raises ValueError when step * eigenvalue is outside (0, 1]")
def test_condition(sub):
    ridge = has(sub, "early_stopping_equivalent_ridge")
    for eigenvalue, step in ((2.0, 0.75), (2.0, 1.5), (3.0, 0.0)):
        try:
            out = ridge(eigenvalue, step, 3)
        except ValueError:
            continue
        fail(f"step * eigenvalue = {eigenvalue * step} should raise ValueError, but the function returned {out}")
    almost(ridge(20.0, 0.05, 4), 0.0, tol=1e-12, label="penalty when one step lands on the answer")


@check("best_stopping_step returns the lowest point of a V-shaped validation curve")
def test_best_step_basic(sub):
    best = has(sub, "best_stopping_step")
    equal(best([5.0, 3.0, 1.0, 2.0, 4.0, 6.0], 3), 2, label="V-shaped curve")
    equal(best([4.0, 3.0, 2.0, 1.0], 2), 3, label="curve that only falls")
    equal(best([2.0, 2.0, 3.0], 5), 0, label="a tie keeps the earlier step")


@check("best_stopping_step stops after `patience` checks without a new best")
def test_best_step_patience(sub):
    best = has(sub, "best_stopping_step")
    losses = [5.0, 4.0, 4.5, 3.0, 3.5, 3.6, 3.7]
    equal(best(losses, 1), 1, label="patience 1 gives up at the first bump")
    equal(best(losses, 2), 3, label="patience 2 survives one bump")
    equal(best([3.0, 3.1, 3.2, 0.1], 2), 0, label="a later dip beyond the patience is never seen")
    equal(best([3.0, 3.1, 0.1], 2), 2, label="a dip within the patience is found")
