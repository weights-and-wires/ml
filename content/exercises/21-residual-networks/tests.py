"""Checks for the residual-network exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def small_params(rng, d=4):
    return {
        "W1": rng.normal(size=(d, d)) * 0.5,
        "b1": rng.normal(size=(d,)) * 0.1,
        "W2": rng.normal(size=(d, d)) * 0.5,
        "b2": rng.normal(size=(d,)) * 0.1,
    }


@check("residual_block_forward matches a direct computation on a tiny example")
def test_forward_direct(sub):
    got = has(sub, "residual_block_forward")
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([[2.0, 0.0], [0.0, 0.5]])
    b2 = np.array([0.1, 0.0])
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])

    y, _ = got(x, params)
    shape(y, (2, 2), label="residual_block_forward output")

    h1 = x @ W1.T + b1
    a1 = np.maximum(h1, 0.0)
    h2 = a1 @ W2.T + b2
    want = x + h2
    close(y, want, atol=1e-10, label="block output")


@check("A zero-initialised residual branch makes the block an exact identity")
def test_zero_branch_identity(sub):
    got = has(sub, "residual_block_forward")
    rng = np.random.default_rng(1)
    d = 5
    params = {
        "W1": rng.normal(size=(d, d)),
        "b1": rng.normal(size=(d,)),
        "W2": np.zeros((d, d)),
        "b2": np.zeros(d),
    }
    x = rng.normal(size=(6, d))
    y, _ = got(x, params)
    close(y, x, atol=1e-12, label="block output with a zeroed last layer")


@check("residual_block_backward matches finite differences for x and every parameter")
def test_backward_finite_difference(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    rng = np.random.default_rng(2)
    d, n = 4, 3
    params = small_params(rng, d)
    x = rng.normal(size=(n, d)) * 0.5
    dY = rng.normal(size=(n, d))

    y, cache = forward(x, params)
    shape(y, (n, d), label="forward output, needed before backward can be checked")
    dx, grads = backward(dY, cache)
    shape(dx, (n, d), label="dx")

    def loss_x(x_probe):
        y_probe, _ = forward(x_probe, params)
        return float(np.sum(dY * y_probe))

    close(dx, finite_difference(loss_x, x, eps=1e-5), rtol=1e-3, atol=1e-4, label="dL/dx")

    for key in ("W1", "b1", "W2", "b2"):

        def loss_param(value, key=key):
            trial = dict(params)
            trial[key] = value
            y_probe, _ = forward(x, trial)
            return float(np.sum(dY * y_probe))

        numeric = finite_difference(loss_param, params[key], eps=1e-5)
        shape(grads[key], params[key].shape, label=f"grad for {key}")
        close(grads[key], numeric, rtol=1e-3, atol=1e-4, label=f"dL/d{key}")


@check("gradient_norm_through_stack keeps the residual stack within an order of magnitude of one at depth 50")
def test_gradient_norm_residual(sub):
    got = has(sub, "gradient_norm_through_stack")
    value = got(50, True)
    if not (0.1 <= value <= 10.0):
        fail(
            "a 50-layer residual stack should keep its gradient norm within an "
            f"order of magnitude of the unit norm it started from, but got {value:.3g}"
        )


@check("gradient_norm_through_stack collapses the plain stack by many orders of magnitude at depth 50")
def test_gradient_norm_plain(sub):
    got = has(sub, "gradient_norm_through_stack")
    value = got(50, False)
    if value > 1e-6:
        fail(
            "a 50-layer plain stack with per-layer Jacobian norm 0.1 should "
            f"collapse to roughly 0.1**50, but got {value:.3g}"
        )


@check("The gap between the residual and plain stacks widens with depth")
def test_gradient_norm_gap_widens(sub):
    got = has(sub, "gradient_norm_through_stack")
    ratio_shallow = got(10, True) / max(got(10, False), 1e-300)
    ratio_deep = got(50, True) / max(got(50, False), 1e-300)
    if not (ratio_deep > ratio_shallow * 1e10):
        fail(
            "the residual-to-plain gradient ratio should grow by many orders of "
            f"magnitude between depth 10 and depth 50, but it went from "
            f"{ratio_shallow:.3g} to {ratio_deep:.3g}"
        )


@check("bottleneck_flops matches the hand-derived multiply-add count")
def test_bottleneck_flops(sub):
    got = has(sub, "bottleneck_flops")
    cases = [
        (64, 16, 64, 8, 8),
        (256, 64, 256, 14, 14),
        (128, 32, 256, 28, 28),
    ]
    for c_in, c_mid, c_out, h, w in cases:
        want = h * w * (c_in * c_mid + 9 * c_mid * c_mid + c_mid * c_out)
        value = got(c_in, c_mid, c_out, h, w)
        if int(value) != want:
            fail(
                f"bottleneck_flops({c_in}, {c_mid}, {c_out}, {h}, {w}) returned "
                f"{value}, expected {want}"
            )


@check("A matched-shape bottleneck is more than fifteen times cheaper than two full-width convolutions")
def test_bottleneck_ratio(sub):
    got = has(sub, "bottleneck_flops")
    c, c_mid, h, w = 256, 64, 14, 14
    bottleneck = got(c, c_mid, c, h, w)
    two_full_width = h * w * 2 * 9 * c * c
    ratio = bottleneck / two_full_width
    if ratio > 1.0 / 15.0:
        fail(
            "a bottleneck with C_mid = C/4 should cost well under a fifteenth "
            f"of two full-width 3x3 convolutions at the same channel count, "
            f"but the ratio was {ratio:.3g}"
        )
