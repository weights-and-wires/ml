"""Checks for the deep-regularisation exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("weight_decay_step matches w - lr * grad - lr * weight_decay * w")
def test_weight_decay_step(sub):
    weight_decay_step = has(sub, "weight_decay_step")
    w = np.array([1.0, -2.0, 0.5])
    grad = np.array([0.1, -0.2, 0.05])
    out = weight_decay_step(w, grad, lr=0.1, weight_decay=0.01)
    close(out, w - 0.1 * grad - 0.1 * 0.01 * w, atol=1e-12, label="weight_decay_step output")


@check("dropout_forward at training=False returns x unchanged with an all-ones mask")
def test_dropout_eval_mode(sub):
    dropout_forward = has(sub, "dropout_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    out, mask = dropout_forward(x, p=0.5, rng=rng, training=False)
    close(out, x, atol=1e-12, label="dropout output at inference")
    close(mask, np.ones_like(x), atol=1e-12, label="mask at inference")


@check("inverted dropout's expected output matches the input over many draws")
def test_dropout_expectation(sub):
    dropout_forward = has(sub, "dropout_forward")
    rng = np.random.default_rng(1)
    x = np.full(2000, 3.0)
    p = 0.3
    totals = np.zeros(2000)
    trials = 400
    for _ in range(trials):
        out, mask = dropout_forward(x, p=p, rng=rng, training=True)
        shape(mask, x.shape, label="mask shape")
        if not np.all((mask == 0.0) | (mask == 1.0)):
            fail("mask should hold only 0s and 1s")
        totals += out
    mean_out = totals / trials
    almost(float(mean_out.mean()), 3.0, tol=0.05, label="mean output across draws")


@check("dropout_backward matches a finite difference with the mask held fixed")
def test_dropout_backward(sub):
    dropout_forward = has(sub, "dropout_forward")
    dropout_backward = has(sub, "dropout_backward")
    p = 0.4
    x = np.linspace(-2.0, 2.0, 12)
    dout = np.linspace(1.0, -1.0, 12)

    out, mask = dropout_forward(x, p=p, rng=np.random.default_rng(7), training=True)
    dx = dropout_backward(dout, mask, p)

    def f(x_):
        rng_fixed = np.random.default_rng(7)  # reproduces the same mask every call
        out_, _ = dropout_forward(x_, p=p, rng=rng_fixed, training=True)
        return float(np.sum(dout * out_))

    close(dx, finite_difference(f, x), atol=1e-6, rtol=1e-4, label="dx vs finite difference")


@check("label_smoothing_grad matches a finite difference of label_smoothing_loss")
def test_label_smoothing_grad(sub):
    label_smoothing_loss = has(sub, "label_smoothing_loss")
    label_smoothing_grad = has(sub, "label_smoothing_grad")
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(5, 4))
    targets = np.array([0, 1, 2, 3, 1])
    eps = 0.15

    grad = label_smoothing_grad(logits, targets, eps)
    shape(grad, logits.shape, label="label_smoothing_grad output")

    f = lambda z: label_smoothing_loss(z, targets, eps)
    close(grad, finite_difference(f, logits), atol=1e-6, rtol=1e-4, label="grad vs finite difference")


@check("the smoothed loss is minimised where p equals the smoothed target, not at a one-hot")
def test_label_smoothing_minimum(sub):
    label_smoothing_loss = has(sub, "label_smoothing_loss")
    label_smoothing_grad = has(sub, "label_smoothing_grad")
    targets = np.array([0])
    eps = 0.2
    k = 4

    # Logits chosen so that softmax(logits) exactly equals the smoothed target.
    p_target = 1.0 - eps + eps / k
    p_other = eps / k
    optimal_logits = np.log(np.array([[p_target, p_other, p_other, p_other]]))
    grad_at_optimum = label_smoothing_grad(optimal_logits, targets, eps)
    if float(np.abs(grad_at_optimum).max()) > 1e-8:
        fail(
            "the gradient should vanish where softmax(logits) equals the smoothed target, "
            f"but its largest entry is {float(np.abs(grad_at_optimum).max()):.3g}"
        )

    confident_logits = np.array([[20.0, 0.0, 0.0, 0.0]])  # softmax is nearly one-hot
    loss_optimum = label_smoothing_loss(optimal_logits, targets, eps)
    loss_confident = label_smoothing_loss(confident_logits, targets, eps)
    if loss_confident <= loss_optimum:
        fail(
            "a near-one-hot prediction should score worse than the smoothed target under "
            f"label smoothing, but got {loss_confident:.4f} against {loss_optimum:.4f}"
        )


@check("early_stopping_equivalent_ridge goes to infinity at 0 steps and to 0 as steps grow")
def test_early_stopping_limits(sub):
    early_stopping_equivalent_ridge = has(sub, "early_stopping_equivalent_ridge")
    gamma_zero_steps = early_stopping_equivalent_ridge(eigenvalue=2.0, step=0.1, iterations=0)
    if not np.isinf(gamma_zero_steps):
        fail(f"expected float('inf') at iterations=0, got {gamma_zero_steps}")
    gamma_many_steps = early_stopping_equivalent_ridge(eigenvalue=2.0, step=0.1, iterations=5000)
    almost(gamma_many_steps, 0.0, tol=1e-6, label="ridge penalty after many steps")


@check("gradient descent for k steps on a quadratic matches the ridge solution the formula predicts")
def test_gd_matches_ridge(sub):
    early_stopping_equivalent_ridge = has(sub, "early_stopping_equivalent_ridge")
    eigenvalues = np.array([0.1, 1.0, 5.0, 20.0])
    w_star = np.array([2.0, -1.5, 0.8, 3.0])
    lr = 0.05
    k = 25

    # Gradient descent on a diagonal quadratic decouples coordinate by coordinate.
    w = np.zeros(4)
    for _ in range(k):
        grad = eigenvalues * (w - w_star)
        w = w - lr * grad

    gammas = np.array(
        [early_stopping_equivalent_ridge(lam, lr, k) for lam in eigenvalues]
    )
    ridge_solution = eigenvalues / (eigenvalues + gammas) * w_star

    close(w, ridge_solution, atol=1e-6, rtol=1e-4, label="gradient descent vs the predicted ridge solution")
