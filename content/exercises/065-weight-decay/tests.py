"""Checks for the weight-decay exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("l2_penalty_grad is the slope of (lam / 2) * sum(w**2), in the shape of w")
def test_penalty_grad(sub):
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
def test_worked_example(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    out = weight_decay_step(np.array([1.0]), np.array([0.1]), lr=0.1, weight_decay=0.01)
    close(out, np.array([0.989]), atol=1e-12, label="weight after one step")


@check("weight_decay_step is a gradient step on the data loss plus the ridge fee")
def test_step_is_ridge_step(sub):
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


@check("with no data gradient, k steps leave w0 * (1 - lr * decay) ** k")
def test_geometric_decay(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    w0 = np.array([1.0, -2.0, 0.5, 4.0])
    w = w0.copy()
    for _ in range(12):
        w = weight_decay_step(w, np.zeros(4), lr=0.5, weight_decay=0.2)
    close(w, w0 * 0.9 ** 12, atol=1e-10, label="weights after 12 steps")


@check("a weight pulled toward 0.6 with decay 0.2 settles at 0.5")
def test_settles(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    w = np.array([1.0])
    for _ in range(200):
        w = weight_decay_step(w, w - 0.6, lr=0.5, weight_decay=0.2)
    close(w, np.array([0.5]), atol=1e-9, label="settled weight, w_star / (1 + decay)")


@check("step_angle matches the angle measured between w and the stepped vector")
def test_angle_measured(sub):
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


@check("halving the length of w roughly doubles the angle for a small step")
def test_angle_halving(sub):
    step_angle = has(sub, "step_angle")
    grad = np.array([0.0, 0.5])
    long = step_angle(np.array([10.0, 0.0]), grad, 0.1)
    short = step_angle(np.array([5.0, 0.0]), grad, 0.1)
    ratio = short / long
    if abs(ratio - 2.0) > 0.01:
        fail(f"expected the angle to about double when w is halved, got a ratio of {ratio:.4f}")
