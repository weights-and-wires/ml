"""Checks for the residual-block forward exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def worked_example():
    params = {
        "W1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b1": np.array([0.0, -1.0]),
        "W2": np.array([[2.0, 0.0], [0.0, 0.5]]),
        "b2": np.array([0.1, 0.0]),
    }
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    return x, params


@check("relu zeroes negative entries, keeps the rest, and leaves its input alone")
def test_relu(sub):
    relu = has(sub, "relu")
    z = np.array([[-2.0, 0.0, 3.0], [1.5, -0.1, -7.0]])
    z_copy = z.copy()
    out = relu(z)
    shape(out, (2, 3), label="relu output")
    close(out, np.array([[0.0, 0.0, 3.0], [1.5, 0.0, 0.0]]), atol=1e-12, label="relu values")
    close(z, z_copy, atol=0.0, label="relu input after the call (it must not change)")


@check("residual_block_forward reproduces the worked example from the chapter")
def test_forward_worked(sub):
    forward = has(sub, "residual_block_forward")
    x, params = worked_example()
    y, _ = forward(x, params)
    shape(y, (2, 2), label="block output")
    close(y, np.array([[3.1, 2.5], [-0.9, 4.0]]), atol=1e-10, label="block output")


@check("A zero last layer makes the block return its input exactly")
def test_zero_branch(sub):
    forward = has(sub, "residual_block_forward")
    rng = np.random.default_rng(1)
    d = 5
    params = {
        "W1": rng.normal(size=(d, d)),
        "b1": rng.normal(size=(d,)),
        "W2": np.zeros((d, d)),
        "b2": np.zeros(d),
    }
    x = rng.normal(size=(6, d))
    y, _ = forward(x, params)
    close(y, x, atol=1e-12, label="block output with W2 = 0 and b2 = 0")


@check("residual_block_forward matches a direct computation and leaves its inputs alone")
def test_forward_random(sub):
    forward = has(sub, "residual_block_forward")
    rng = np.random.default_rng(2)
    n, d = 7, 3
    params = {
        "W1": rng.normal(size=(d, d)),
        "b1": rng.normal(size=(d,)),
        "W2": rng.normal(size=(d, d)),
        "b2": rng.normal(size=(d,)),
    }
    x = rng.normal(size=(n, d))
    x_copy = x.copy()
    params_copy = {k: v.copy() for k, v in params.items()}

    y, _ = forward(x, params)
    h1 = x @ params["W1"].T + params["b1"]
    want = x + np.maximum(h1, 0.0) @ params["W2"].T + params["b2"]
    shape(y, (n, d), label="block output")
    close(y, want, atol=1e-10, label="block output")
    close(x, x_copy, atol=0.0, label="x after the call")
    for key in params:
        close(params[key], params_copy[key], atol=0.0, label=f"params[{key!r}] after the call")


@check("The cache holds x, W1, W2, h1, and a1 in that order")
def test_cache(sub):
    forward = has(sub, "residual_block_forward")
    x, params = worked_example()
    _, cache = forward(x, params)
    if not isinstance(cache, tuple) or len(cache) != 5:
        fail(f"cache should be a tuple of 5 entries (x, W1, W2, h1, a1), got {type(cache).__name__} of length {len(cache) if hasattr(cache, '__len__') else '?'}")
    close(cache[0], x, atol=1e-12, label="cache[0] (x)")
    close(cache[1], params["W1"], atol=1e-12, label="cache[1] (W1)")
    close(cache[2], params["W2"], atol=1e-12, label="cache[2] (W2)")
    close(cache[3], np.array([[1.0, 1.0], [-1.0, 2.0]]), atol=1e-12, label="cache[3] (h1)")
    close(cache[4], np.array([[1.0, 1.0], [0.0, 2.0]]), atol=1e-12, label="cache[4] (a1)")


@check("identity_gap is exactly 0 for a residual block with a zero branch, and max|x| for a plain block")
def test_gap_zero(sub):
    gap = has(sub, "identity_gap")
    rng = np.random.default_rng(4)
    d = 4
    params = {
        "W1": rng.normal(size=(d, d)),
        "b1": rng.normal(size=(d,)),
        "W2": np.zeros((d, d)),
        "b2": np.zeros(d),
    }
    x = rng.normal(size=(5, d))
    g_res = gap(x, params, True)
    g_plain = gap(x, params, False)
    if abs(g_res) > 1e-12:
        fail(f"a residual block with a zero last layer should have gap 0, got {g_res}")
    want = float(np.max(np.abs(x)))
    if abs(g_plain - want) > 1e-10:
        fail(f"a plain block with a zero last layer outputs all zeros, so its gap should be max|x| = {want:.6f}, got {g_plain}")


@check("With small random weights the residual block stays close to its input and the plain block does not")
def test_gap_small_weights(sub):
    gap = has(sub, "identity_gap")
    rng = np.random.default_rng(3)
    d, n = 8, 5
    params = {
        "W1": rng.normal(size=(d, d)) * 0.05,
        "b1": np.zeros(d),
        "W2": rng.normal(size=(d, d)) * 0.05,
        "b2": np.zeros(d),
    }
    x = rng.normal(size=(n, d))
    g_res = gap(x, params, True)
    g_plain = gap(x, params, False)
    if not g_res < 0.1:
        fail(f"a residual block with weights of scale 0.05 should move entries by under 0.1, got {g_res:.4f}")
    if not g_plain > 1.0:
        fail(f"a plain block with weights of scale 0.05 should output nearly zero, so its gap should exceed 1, got {g_plain:.4f}")
