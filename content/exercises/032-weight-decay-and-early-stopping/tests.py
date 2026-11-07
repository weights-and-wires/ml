"""Checks for the weight decay and early stopping exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape


@check("l2_penalty_grad is the slope of (lam / 2) * sum(w**2), in the shape of w")
def test_l2_penalty_grad_is_the_slope_of_lam_2_su(sub):
    l2_penalty_grad = has(sub, "l2_penalty_grad")
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 2))
    lam = 0.7
    out = l2_penalty_grad(w, lam)
    shape(out, w.shape, label="l2_penalty_grad output")
    fee = lambda v: 0.5 * lam * float(np.sum(v ** 2))
    close(out, finite_difference(fee, w), atol=1e-6, rtol=1e-5, label="slope vs finite difference")
    close(l2_penalty_grad(w, 0.0), np.zeros_like(w), atol=1e-12, label="slope when lam is 0")


@check("weight_decay_step turns w = 1 into 0.989 on the worked example")
def test_weight_decay_step_turns_w_1_into_0_989_o(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    out = weight_decay_step(np.array([1.0]), np.array([0.1]), lr=0.1, weight_decay=0.01)
    close(out, np.array([0.989]), atol=1e-12, label="weight after one step")


@check("weight_decay_step is a gradient step on the data loss plus the ridge fee")
def test_weight_decay_step_is_a_gradient_step_on(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 3))
    y = rng.normal(size=6)
    w = rng.normal(size=3)
    lr, lam = 0.05, 0.3
    total = lambda v: 0.5 * float(np.sum((X @ v - y) ** 2)) + 0.5 * lam * float(np.sum(v ** 2))
    want = w - lr * finite_difference(total, w)
    got = weight_decay_step(w, X.T @ (X @ w - y), lr, lam)
    close(got, want, atol=1e-6, rtol=1e-5, label="step vs finite-difference step on the total loss")


@check("with no data gradient k steps leave w0 * (1 - lr * decay) ** k, and a weight pulled toward 0.6 settles at 0.5")
def test_with_no_data_gradient_k_steps_leave_w0_1(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    w0 = np.array([1.0, -2.0, 0.5, 4.0])
    w = w0.copy()
    for _ in range(12):
        w = weight_decay_step(w, np.zeros(4), lr=0.5, weight_decay=0.2)
    close(w, w0 * 0.9 ** 12, atol=1e-10, label="weights after 12 steps")
    weight_decay_step = has(sub, "weight_decay_step")
    w = np.array([1.0])
    for _ in range(200):
        w = weight_decay_step(w, w - 0.6, lr=0.5, weight_decay=0.2)
    close(w, np.array([0.5]), atol=1e-9, label="settled weight, w_star / (1 + decay)")


@check("step_angle matches the measured angle, and halving the length of w roughly doubles it")
def test_step_angle_matches_the_measured_angle_an(sub):
    step_angle = has(sub, "step_angle")
    w = np.array([3.0, 0.0, 0.0])
    grad = np.array([0.0, 0.4, 0.0])
    lr = 2.0
    moved = w - lr * grad
    want = float(np.arccos(w @ moved / (np.linalg.norm(w) * np.linalg.norm(moved))))
    got = step_angle(w, grad, lr)
    if not isinstance(got, float):
        fail(f"step_angle should return a float, got {type(got).__name__}")
    almost(got, want, tol=1e-9, label="angle in radians")
    step_angle = has(sub, "step_angle")
    grad = np.array([0.0, 0.5])
    long = step_angle(np.array([10.0, 0.0]), grad, 0.1)
    short = step_angle(np.array([5.0, 0.0]), grad, 0.1)
    ratio = short / long
    if abs(ratio - 2.0) > 0.01:
        fail(f"expected the angle to about double when w is halved, got a ratio of {ratio:.4f}")


@check("gd_quadratic_1d matches (1 - (1 - step * eigenvalue) ** k) * w_star, even when a step overshoots")
def test_gd_quadratic_1d_matches_1_1_step_eigenva(sub):
    gd_quadratic_1d = has(sub, "gd_quadratic_1d")
    for eigenvalue, step, w_star in ((2.0, 0.1, 1.0), (0.5, 0.05, -3.0), (1.0, 1.5, 2.0)):
        for k in (0, 1, 2, 10, 25):
            want = (1.0 - (1.0 - step * eigenvalue) ** k) * w_star
            got = gd_quadratic_1d(eigenvalue, step, w_star, k)
            almost(got, want, tol=1e-9, label=f"weight at c={eigenvalue}, step={step}, k={k}")
    almost(gd_quadratic_1d(2.0, 0.1, 1.0, 10), 0.89263, tol=1e-5, label="worked example at k = 10")


@check("early_stopping_equivalent_ridge is infinite at 0 steps, 8 at 1 step, and 0.24058 at 10 steps, and falls as steps grow")
def test_early_stopping_equivalent_ridge_is_infin(sub):
    ridge = has(sub, "early_stopping_equivalent_ridge")
    got0 = ridge(2.0, 0.1, 0)
    if not np.isinf(got0):
        fail(f"expected float('inf') at 0 steps, got {got0}")
    almost(ridge(2.0, 0.1, 1), 8.0, tol=1e-9, label="penalty at k = 1")
    almost(ridge(2.0, 0.1, 10), 0.24058, tol=1e-5, label="penalty at k = 10")
    ridge = has(sub, "early_stopping_equivalent_ridge")
    values = [ridge(2.0, 0.1, k) for k in range(1, 40)]
    if any(b >= a for a, b in zip(values, values[1:])):
        fail("the penalty should fall at every extra step, but it did not")
    almost(ridge(2.0, 0.1, 5000), 0.0, tol=1e-9, label="penalty after many steps")
    flat, middle, steep = (ridge(c, 0.1, 25) for c in (0.1, 1.0, 5.0))
    if not (flat > middle > steep):
        fail(f"at 25 steps expected penalties to fall as curvature rises, got {flat:.4g}, {middle:.4g}, {steep:.4g}")


@check("k steps of gradient descent land on the ridge solution that the penalty predicts")
def test_k_steps_of_gradient_descent_land_on_the(sub):
    gd_quadratic_1d = has(sub, "gd_quadratic_1d")
    ridge = has(sub, "early_stopping_equivalent_ridge")
    for eigenvalue, step, w_star in ((2.0, 0.1, 1.0), (0.1, 0.05, 4.0), (5.0, 0.1, -1.5), (20.0, 0.05, 3.0)):
        for k in (1, 5, 25):
            lam = ridge(eigenvalue, step, k)
            ridge_w = eigenvalue / (eigenvalue + lam) * w_star
            got = gd_quadratic_1d(eigenvalue, step, w_star, k)
            almost(ridge_w, got, tol=1e-8, label=f"ridge solution vs gradient descent at c={eigenvalue}, k={k}")


@check("early_stopping_equivalent_ridge raises ValueError when step * eigenvalue is outside (0, 1]")
def test_early_stopping_equivalent_ridge_raises_v(sub):
    ridge = has(sub, "early_stopping_equivalent_ridge")
    for eigenvalue, step in ((2.0, 0.75), (2.0, 1.5), (3.0, 0.0)):
        try:
            out = ridge(eigenvalue, step, 3)
        except ValueError:
            continue
        fail(f"step * eigenvalue = {eigenvalue * step} should raise ValueError, but the function returned {out}")
    almost(ridge(20.0, 0.05, 4), 0.0, tol=1e-12, label="penalty when one step lands on the answer")


@check("best_stopping_step finds the lowest point of a V-shaped curve and respects patience on a jumpy one")
def test_best_stopping_step_finds_the_lowest_poin(sub):
    best = has(sub, "best_stopping_step")
    equal(best([5.0, 3.0, 1.0, 2.0, 4.0, 6.0], 3), 2, label="V-shaped curve")
    equal(best([4.0, 3.0, 2.0, 1.0], 2), 3, label="curve that only falls")
    equal(best([2.0, 2.0, 3.0], 5), 0, label="a tie keeps the earlier step")
    best = has(sub, "best_stopping_step")
    losses = [5.0, 4.0, 4.5, 3.0, 3.5, 3.6, 3.7]
    equal(best(losses, 1), 1, label="patience 1 gives up at the first bump")
    equal(best(losses, 2), 3, label="patience 2 survives one bump")
    equal(best([3.0, 3.1, 3.2, 0.1], 2), 0, label="a later dip beyond the patience is never seen")
    equal(best([3.0, 3.1, 0.1], 2), 2, label="a dip within the patience is found")
