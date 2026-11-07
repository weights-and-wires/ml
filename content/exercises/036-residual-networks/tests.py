"""Checks for the residual networks exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape


def worked_example():
    params = {
        "W1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b1": np.array([0.0, -1.0]),
        "W2": np.array([[2.0, 0.0], [0.0, 0.5]]),
        "b2": np.array([0.1, 0.0]),
    }
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    return x, params


def random_case(seed, n=3, d=4):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(size=(d, d)) * 0.5,
        "b1": rng.normal(size=(d,)) * 0.1,
        "W2": rng.normal(size=(d, d)) * 0.5,
        "b2": rng.normal(size=(d,)) * 0.1,
    }
    x = rng.normal(size=(n, d)) * 0.5
    dY = rng.normal(size=(n, d))
    return x, params, dY


@check("residual_block_forward reproduces the worked example, and its cache holds x, W1, W2, h1, and a1 in that order")
def test_forward_worked_and_cache(sub):
    forward = has(sub, "residual_block_forward")
    x, params = worked_example()
    y, _ = forward(x, params)
    shape(y, (2, 2), label="block output")
    close(y, np.array([[3.1, 2.5], [-0.9, 4.0]]), atol=1e-10, label="block output")
    x, params = worked_example()
    _, cache = forward(x, params)
    if not isinstance(cache, tuple) or len(cache) != 5:
        fail(f"cache should be a tuple of 5 entries (x, W1, W2, h1, a1), got {type(cache).__name__} of length {len(cache) if hasattr(cache, '__len__') else '?'}")
    close(cache[0], x, atol=1e-12, label="cache[0] (x)")
    close(cache[1], params["W1"], atol=1e-12, label="cache[1] (W1)")
    close(cache[2], params["W2"], atol=1e-12, label="cache[2] (W2)")
    close(cache[3], np.array([[1.0, 1.0], [-1.0, 2.0]]), atol=1e-12, label="cache[3] (h1)")
    close(cache[4], np.array([[1.0, 1.0], [0.0, 2.0]]), atol=1e-12, label="cache[4] (a1)")


@check("A zero last layer returns the input exactly, and the forward pass matches a direct computation and leaves its inputs alone")
def test_forward_zero_and_random(sub):
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


@check("identity_gap separates the two block types, with zero weights and with small random weights")
def test_identity_gap(sub):
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


@check("residual_block_backward reproduces every gradient in the worked example")
def test_backward_worked(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    x, params = worked_example()
    _, cache = forward(x, params)
    dx, grads = backward(np.ones((2, 2)), cache)
    shape(dx, (2, 2), label="dx")
    close(dx, np.array([[3.0, 1.5], [1.0, 1.5]]), atol=1e-10, label="dx")
    close(grads["W2"], np.array([[1.0, 3.0], [1.0, 3.0]]), atol=1e-10, label="dW2")
    close(grads["b2"], np.array([2.0, 2.0]), atol=1e-10, label="db2")
    close(grads["W1"], np.array([[2.0, 4.0], [0.0, 2.5]]), atol=1e-10, label="dW1")
    close(grads["b1"], np.array([2.0, 1.0]), atol=1e-10, label="db1")


@check("residual_block_backward matches finite differences for x and every parameter")
def test_backward_fd(sub):
    forward = has(sub, "residual_block_forward")
    backward = has(sub, "residual_block_backward")
    x, params, dY = random_case(2)
    y, cache = forward(x, params)
    dx, grads = backward(dY, cache)
    shape(dx, x.shape, label="dx")

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

        shape(grads[key], params[key].shape, label=f"grad for {key}")
        numeric = finite_difference(loss_param, params[key], eps=1e-5)
        close(grads[key], numeric, rtol=1e-3, atol=1e-4, label=f"dL/d{key}")


@check("orthogonal_matrix returns Q with Q.T @ Q equal to the identity, keeps lengths, and differs between draws")
def test_orthogonal(sub):
    orthogonal_matrix = has(sub, "orthogonal_matrix")
    rng = np.random.default_rng(0)
    for d in (2, 5, 16):
        q = orthogonal_matrix(d, rng)
        shape(q, (d, d), label=f"orthogonal_matrix({d}) output")
        close(q.T @ q, np.eye(d), atol=1e-10, label=f"Q.T @ Q for d = {d}")
    rng = np.random.default_rng(1)
    q1 = orthogonal_matrix(8, rng)
    q2 = orthogonal_matrix(8, rng)
    v = rng.normal(size=8)
    almost(np.linalg.norm(v @ q1), np.linalg.norm(v), tol=1e-10, label="length of v @ Q")
    if np.allclose(q1, q2):
        fail("two draws from the same generator gave the same matrix; draw from rng each time")


@check("scalar_gradient_factor is the toy model, and in a real stack the plain gradient shrinks by exactly 0.1 per layer while the residual one stays alive")
def test_stack(sub):
    got = has(sub, "gradient_norm_through_stack")
    factor = has(sub, "scalar_gradient_factor")
    almost(factor(0.1, 50, False), 0.1**50, tol=1e-60, label="plain, J = 0.1, 50 layers")
    almost(factor(0.1, 3, True), 1.1**3, tol=1e-12, label="residual, J = 0.1, 3 layers")
    almost(factor(0.1, 50, True), 1.1**50, tol=1e-6, label="residual, J = 0.1, 50 layers")
    almost(got(0, True), 1.0, tol=1e-12, label="zero-depth residual stack")
    almost(got(0, False), 1.0, tol=1e-12, label="zero-depth plain stack")
    for depth in (1, 10, 50):
        value = got(depth, False)
        want = 0.1**depth
        if not abs(value - want) <= 1e-6 * want:
            fail(
                f"a plain stack of {depth} layers, each 0.1 times an orthogonal matrix, "
                f"should leave a gradient of length 0.1**{depth} = {want:.3e}, but got {value:.3e}. "
                "If the number is close but not equal, the matrices are not orthogonal."
            )
    value = got(50, True)
    if not (0.1 <= value <= 10.0):
        fail(f"a 50-layer residual stack should keep the gradient length between 0.1 and 10, but got {value:.3g}")
    toy = factor(0.1, 50, True)
    if not value < toy / 10:
        fail(
            f"the scalar model predicts growth to {toy:.3g}, but notes that point at angles "
            f"should keep the real length far below that; your stack gave {value:.3g}"
        )


@check("conv_flops is a product of height, width, both channel counts, and the kernel area, and returns an int")
def test_conv(sub):
    conv_flops = has(sub, "conv_flops")
    equal(conv_flops(4, 4, 1, 2, 2), 64, label="1x1 convolution, 4 -> 4 channels, 2x2 picture")
    equal(conv_flops(4, 4, 3, 2, 2), 576, label="3x3 convolution, 4 -> 4 channels, 2x2 picture")
    cases = [(3, 8, 3, 32, 32), (256, 256, 3, 14, 14), (64, 128, 1, 28, 14), (16, 16, 5, 7, 9)]
    for c_in, c_out, k, h, w in cases:
        want = h * w * c_in * c_out * k * k
        got = conv_flops(c_in, c_out, k, h, w)
        if int(got) != want:
            fail(f"conv_flops({c_in}, {c_out}, {k}, {h}, {w}) returned {got}, expected {want}")
    if not isinstance(conv_flops(8, 8, 3, 4, 4), int):
        fail("conv_flops should return an int")


@check("bottleneck_flops matches the hand-computed count, including 256 -> 64 -> 256 at 14x14")
def test_bottleneck(sub):
    bottleneck_flops = has(sub, "bottleneck_flops")
    equal(bottleneck_flops(256, 64, 256, 14, 14), 13_647_872, label="256 -> 64 -> 256 at 14x14")
    cases = [(64, 16, 64, 8, 8), (128, 32, 256, 28, 28), (10, 3, 7, 5, 6), (16, 128, 16, 1, 1)]
    for c_in, c_mid, c_out, h, w in cases:
        want = h * w * (c_in * c_mid + 9 * c_mid * c_mid + c_mid * c_out)
        got = bottleneck_flops(c_in, c_mid, c_out, h, w)
        if int(got) != want:
            fail(f"bottleneck_flops({c_in}, {c_mid}, {c_out}, {h}, {w}) returned {got}, expected {want}; the middle convolution is 3x3, so it carries a factor of 9")
    if not isinstance(bottleneck_flops(8, 2, 8, 4, 4), int):
        fail("bottleneck_flops should return an int")


@check("bottleneck_savings is about 17 for 256 channels at 14x14, and 288/17 at every width divisible by 4")
def test_savings(sub):
    savings = has(sub, "bottleneck_savings")
    value = savings(256, 14, 14)
    almost(value, 231_211_008 / 13_647_872, tol=1e-9, label="savings at 256 channels, 14x14")
    if not (16.0 < value < 18.0):
        fail(f"the bottleneck should be about 17 times cheaper, got {value:.3f}")
    for c, h, w in [(64, 8, 8), (128, 28, 28), (256, 7, 7), (512, 14, 3)]:
        almost(savings(c, h, w), 288 / 17, tol=1e-9, label=f"savings at c={c}, {h}x{w}")
