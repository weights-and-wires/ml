"""Checks for the label-smoothing exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("smooth_targets gives [0.9333, 0.0333, 0.0333] for class 0 with K = 3 and eps = 0.1")
def test_smooth_targets(sub):
    smooth_targets = has(sub, "smooth_targets")
    out = smooth_targets(np.array([0, 2]), 3, 0.1)
    shape(out, (2, 3), label="smooth_targets output")
    want = np.array([[0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3], [0.1 / 3, 0.1 / 3, 0.9 + 0.1 / 3]])
    close(out, want, atol=1e-12, label="smoothed targets")
    close(out.sum(axis=1), np.ones(2), atol=1e-12, label="row sums")
    close(smooth_targets(np.array([1, 0]), 2, 0.0), np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12, label="eps = 0 gives one-hot")


@check("label_smoothing_loss is 0.3729 on the worked row, and 0.2395 with eps = 0")
def test_loss_worked_row(sub):
    loss = has(sub, "label_smoothing_loss")
    logits = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    got = loss(logits, targets, 0.1)
    if not isinstance(got, float):
        fail(f"label_smoothing_loss should return a float, got {type(got).__name__}")
    almost(got, 0.372878, tol=1e-5, label="loss with eps = 0.1")
    almost(loss(logits, targets, 0.0), 0.239545, tol=1e-5, label="loss with eps = 0")


@check("label_smoothing_loss is the mean over rows")
def test_loss_mean(sub):
    loss = has(sub, "label_smoothing_loss")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 5))
    targets = np.array([0, 3, 1, 4])
    singles = [loss(logits[i:i + 1], targets[i:i + 1], 0.2) for i in range(4)]
    almost(loss(logits, targets, 0.2), float(np.mean(singles)), tol=1e-10, label="batch loss vs mean of single-row losses")


@check("label_smoothing_loss stays finite at scores of 800, where it equals 2 * (eps / K) * 800")
def test_loss_stable(sub):
    loss = has(sub, "label_smoothing_loss")
    logits = np.array([[800.0, 0.0, 0.0]])
    got = loss(logits, np.array([0]), 0.1)
    if not np.isfinite(got):
        fail(f"loss is not finite at scores of 800: {got}")
    almost(got, 2 * (0.1 / 3) * 800, tol=1e-6, label="loss at a gap of 800")
    almost(loss(logits, np.array([0]), 0.0), 0.0, tol=1e-9, label="loss at a gap of 800 with eps = 0")


@check("label_smoothing_grad matches a finite difference of label_smoothing_loss")
def test_grad_finite_difference(sub):
    loss = has(sub, "label_smoothing_loss")
    grad = has(sub, "label_smoothing_grad")
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(5, 4))
    targets = np.array([0, 1, 2, 3, 1])
    eps = 0.15
    got = grad(logits, targets, eps)
    shape(got, logits.shape, label="label_smoothing_grad output")
    close(got, finite_difference(lambda z: loss(z, targets, eps), logits), atol=1e-6, rtol=1e-4, label="grad vs finite difference")


@check("label_smoothing_grad on the worked row is [-0.1463, 0.0732, 0.0732] and vanishes where p equals the target")
def test_grad_values(sub):
    grad = has(sub, "label_smoothing_grad")
    got = grad(np.array([[2.0, 0.0, 0.0]]), np.array([0]), 0.1)
    close(got, np.array([[-0.146347, 0.073174, 0.073174]]), atol=1e-5, label="gradient on the worked row")
    close(got.sum(axis=1), np.zeros(1), atol=1e-12, label="gradient row sum")
    eps, k = 0.2, 4
    optimal = np.log(np.array([[1 - eps + eps / k, eps / k, eps / k, eps / k]]))
    at_optimum = grad(optimal, np.array([0]), eps)
    if float(np.abs(at_optimum).max()) > 1e-9:
        fail(f"the gradient should vanish where softmax equals the smoothed target, but its largest entry is {float(np.abs(at_optimum).max()):.3g}")


@check("the loss over score gaps is lowest near ln 28 = 3.33, and the one-hot loss never turns up")
def test_minimum(sub):
    loss = has(sub, "label_smoothing_loss")
    gaps = np.linspace(0.0, 8.0, 801)
    targets = np.array([0])
    smoothed = np.array([loss(np.array([[g, 0.0, 0.0]]), targets, 0.1) for g in gaps])
    best = float(gaps[int(np.argmin(smoothed))])
    almost(best, float(np.log(28.0)), tol=0.02, label="gap with the lowest smoothed loss")
    plain = np.array([loss(np.array([[g, 0.0, 0.0]]), targets, 0.0) for g in gaps])
    if np.any(np.diff(plain) >= 0):
        fail("with eps = 0 the loss should keep falling as the gap grows")
