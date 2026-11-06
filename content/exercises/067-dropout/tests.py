"""Checks for the dropout exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("dropout_mask holds only 0s and 1s, keeps about a 1 - p share, and repeats under the same seed")
def test_mask(sub):
    dropout_mask = has(sub, "dropout_mask")
    p = 0.3
    mask = dropout_mask((400, 500), p, np.random.default_rng(0))
    shape(mask, (400, 500), label="mask shape")
    if not np.all((mask == 0.0) | (mask == 1.0)):
        fail("the mask should hold only 0.0 and 1.0")
    almost(float(mask.mean()), 1.0 - p, tol=0.01, label="share of units kept")
    again = dropout_mask((400, 500), p, np.random.default_rng(0))
    close(again, mask, atol=0.0, label="mask with the same seed")
    other = dropout_mask((400, 500), p, np.random.default_rng(1))
    if np.array_equal(other, mask):
        fail("different seeds should give different masks")


@check("dropout_forward at training=False returns x unchanged with an all-ones mask")
def test_eval_mode(sub):
    dropout_forward = has(sub, "dropout_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    out, mask = dropout_forward(x, p=0.5, rng=rng, training=False)
    close(out, x, atol=1e-12, label="output at test time")
    close(mask, np.ones_like(x), atol=1e-12, label="mask at test time")


@check("dropout_forward builds its mask with dropout_mask and returns x * mask / (1 - p)")
def test_forward_scaling(sub):
    dropout_forward = has(sub, "dropout_forward")
    dropout_mask = has(sub, "dropout_mask")
    x = np.array([[2.0, 4.0, 6.0, 8.0], [1.0, -3.0, 5.0, 7.0]])
    # p = 0.5 cannot tell 1 - p from p, so also try p = 0.25.
    for p in (0.5, 0.25):
        out, mask = dropout_forward(x, p, np.random.default_rng(5), True)
        shape(out, x.shape, label="output shape")
        close(mask, dropout_mask(x.shape, p, np.random.default_rng(5)), atol=0.0, label=f"mask from the same seed at p={p}")
        close(out, x * mask / (1.0 - p), atol=1e-12, label=f"output vs x * mask / (1 - p) at p={p}")
        if np.any(out[mask == 0.0] != 0.0):
            fail("dropped units should output exactly 0")


@check("the average output over many draws matches the input, so the layer sum averages 20")
def test_expectation(sub):
    dropout_forward = has(sub, "dropout_forward")
    rng = np.random.default_rng(1)
    x = np.array([2.0, 4.0, 6.0, 8.0])
    trials = 4000
    total = np.zeros(4)
    for _ in range(trials):
        out, _ = dropout_forward(x, 0.5, rng, True)
        total += out
    mean = total / trials
    close(mean, x, atol=0.6, label="mean output per unit, near [2, 4, 6, 8]")
    almost(float(mean.sum()), 20.0, tol=1.0, label="mean layer sum")


@check("dropout_backward on the worked example turns dout = [1, 1, 1, 1] into [2, 0, 2, 2]")
def test_backward_example(sub):
    dropout_backward = has(sub, "dropout_backward")
    dx = dropout_backward(np.ones(4), np.array([1.0, 0.0, 1.0, 1.0]), 0.5)
    close(dx, np.array([2.0, 0.0, 2.0, 2.0]), atol=1e-12, label="dx for the worked example")
    dx = dropout_backward(np.ones(4), np.array([1.0, 0.0, 1.0, 1.0]), 0.25)
    close(dx, np.array([4 / 3, 0.0, 4 / 3, 4 / 3]), atol=1e-12, label="dx at p = 0.25 (divide by 1 - p, not p)")


@check("dropout_backward matches a finite difference with the mask held fixed")
def test_backward_finite_difference(sub):
    dropout_forward = has(sub, "dropout_forward")
    dropout_backward = has(sub, "dropout_backward")
    p = 0.4
    x = np.linspace(-2.0, 2.0, 12).reshape(3, 4)
    dout = np.linspace(1.0, -1.0, 12).reshape(3, 4)

    _, mask = dropout_forward(x, p, np.random.default_rng(7), True)
    dx = dropout_backward(dout, mask, p)

    def f(x_):
        out_, _ = dropout_forward(x_, p, np.random.default_rng(7), True)  # same seed, same mask
        return float(np.sum(dout * out_))

    close(dx, finite_difference(f, x), atol=1e-6, rtol=1e-4, label="dx vs finite difference")
