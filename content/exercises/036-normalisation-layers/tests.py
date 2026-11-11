"""Checks for the normalisation layers exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("normalise_group turns [1, 2, 3] into about [-1.2247, 0, 1.2247], ignores shift and scale, and stays finite on a constant group")
def test_normalise_group(sub):
    normalise_group = has(sub, "normalise_group")
    out = normalise_group(np.array([1.0, 2.0, 3.0]), eps=0.0)
    close(out, np.array([-1.224744871, 0.0, 1.224744871]), atol=1e-6, label="[1, 2, 3] with eps = 0")
    out = normalise_group(np.array([1.0, 2.0, 3.0]))
    close(out, np.array([-1.2247, 0.0, 1.2247]), atol=1e-3, label="[1, 2, 3] with the default eps")

    rng = np.random.default_rng(0)
    x = rng.normal(loc=4.0, scale=3.0, size=50)
    out = normalise_group(x)
    almost(out.mean(), 0.0, tol=1e-9, label="mean")
    almost(out.var(), 1.0, tol=1e-3, label="variance")
    close(normalise_group(10.0 * x + 37.5), out, atol=1e-4, label="after scaling by 10 and shifting by 37.5")

    const = normalise_group(np.array([5.0, 5.0, 5.0]))
    if not np.all(np.isfinite(const)):
        fail(f"a constant group gave {const}; adding eps to the variance keeps the division finite")
    close(const, np.zeros(3), atol=1e-12, label="[5, 5, 5]")


@check("batchnorm_forward gives each column mean 0 and variance 1, and applies gamma and beta")
def test_batchnorm_forward(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(500, 6))
    out, cache = batchnorm_forward(x, np.ones(6), np.zeros(6))
    shape(out, x.shape, label="batchnorm_forward output")
    close(out.mean(axis=0), np.zeros(6), atol=1e-6, label="column means")
    close(out.var(axis=0), np.ones(6), atol=1e-3, label="column variances")
    gamma = np.linspace(0.5, 2.0, 6)
    beta = np.linspace(-1.0, 1.0, 6)
    out2, _ = batchnorm_forward(x, gamma, beta)
    close(out2.mean(axis=0), beta, atol=1e-6, label="column means with beta")
    close(out2.std(axis=0), gamma, atol=1e-2, label="column spreads with gamma")
    if not isinstance(cache, tuple) or len(cache) != 4:
        fail("the cache should be the tuple (xhat, istd, gamma, axis)")


@check("layernorm_forward gives each row mean 0 and variance 1, and applies gamma and beta")
def test_layernorm_forward(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    rng = np.random.default_rng(1)
    x = rng.normal(loc=-2.0, scale=4.0, size=(50, 200))
    out, cache = layernorm_forward(x, np.ones(200), np.zeros(200))
    shape(out, x.shape, label="layernorm_forward output")
    close(out.mean(axis=1), np.zeros(50), atol=1e-6, label="row means")
    close(out.var(axis=1), np.ones(50), atol=1e-3, label="row variances")
    gamma = rng.normal(size=200) + 2.0
    beta = rng.normal(size=200)
    out2, _ = layernorm_forward(x, gamma, beta)
    close(out2, gamma * out + beta, atol=1e-9, label="output equals gamma * xhat + beta")
    if not isinstance(cache, tuple) or len(cache) != 4:
        fail("the cache should be the tuple (xhat, istd, gamma, axis)")


@check("A batch of one gives beta under batch norm and not under layer norm, and layer norm ignores a shifted example")
def test_batch_of_one_and_shift(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    layernorm_forward = has(sub, "layernorm_forward")
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    beta = np.array([0.5, -1.0, 2.0, 0.0])
    out, _ = batchnorm_forward(x, np.ones(4), beta)
    close(out, beta[None, :], atol=1e-9, label="batch norm of a single example")
    ln, _ = layernorm_forward(x, np.ones(4), np.zeros(4))
    close(ln, np.array([[-1.3416, -0.4472, 0.4472, 1.3416]]), atol=1e-3, label="layer norm of the same example")

    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 6))
    gamma, beta = np.ones(6), np.zeros(6)
    shifted = x.copy()
    shifted[0] += 37.5
    ln_before, _ = layernorm_forward(x, gamma, beta)
    ln_after, _ = layernorm_forward(shifted, gamma, beta)
    close(ln_after[0], ln_before[0], atol=1e-6, label="layer norm's row 0 after the shift")
    bn_before, _ = batchnorm_forward(x, gamma, beta)
    bn_after, _ = batchnorm_forward(shifted, gamma, beta)
    moved = float(np.abs(bn_after[0] - bn_before[0]).max())
    if moved < 1e-3:
        fail(
            "batch norm normalises down each column, so shifting one example moves the column "
            f"statistics and that example's output, but it moved only {moved:.3g}"
        )


@check("affine_param_grads sums dout and dout * xhat over the examples")
def test_affine_param_grads(sub):
    affine_param_grads = has(sub, "affine_param_grads")
    xhat = np.array([[1.0, 2.0], [3.0, 4.0]])
    dout = np.array([[1.0, 0.0], [1.0, 1.0]])
    dgamma, dbeta = affine_param_grads(dout, xhat)
    shape(dgamma, (2,), label="dgamma")
    shape(dbeta, (2,), label="dbeta")
    close(dbeta, np.array([2.0, 1.0]), atol=1e-12, label="dbeta")
    close(dgamma, np.array([4.0, 4.0]), atol=1e-12, label="dgamma")


@check("batchnorm_backward matches finite differences for x, gamma, and beta")
def test_batchnorm_backward(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    batchnorm_backward = has(sub, "batchnorm_backward")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 4))
    gamma = rng.normal(size=4) + 1.5
    beta = rng.normal(size=4)
    dout = rng.normal(size=(6, 4))

    _, cache = batchnorm_forward(x, gamma, beta)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    shape(dx, x.shape, label="dx")
    shape(dgamma, gamma.shape, label="dgamma")
    shape(dbeta, beta.shape, label="dbeta")

    f_x = lambda x_: float(np.sum(dout * batchnorm_forward(x_, gamma, beta)[0]))
    f_gamma = lambda g_: float(np.sum(dout * batchnorm_forward(x, g_, beta)[0]))
    f_beta = lambda b_: float(np.sum(dout * batchnorm_forward(x, gamma, b_)[0]))

    close(dx, finite_difference(f_x, x), atol=1e-4, rtol=1e-3, label="dx vs finite difference")
    close(dgamma, finite_difference(f_gamma, gamma), atol=1e-4, rtol=1e-3, label="dgamma vs finite difference")
    close(dbeta, finite_difference(f_beta, beta), atol=1e-4, rtol=1e-3, label="dbeta vs finite difference")


@check("layernorm_backward reproduces the hand example for [1, 2, 3] and matches finite differences")
def test_layernorm_backward(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    layernorm_backward = has(sub, "layernorm_backward")
    x = np.array([[1.0, 2.0, 3.0]])
    _, cache = layernorm_forward(x, np.ones(3), np.zeros(3))
    dx, dgamma, dbeta = layernorm_backward(np.array([[1.0, 0.0, 0.0]]), cache)
    close(dx, np.array([[0.2041, -0.4082, 0.2041]]), atol=1e-3, label="dx for dout = [1, 0, 0]")
    close(dgamma, np.array([-1.2247, 0.0, 0.0]), atol=1e-3, label="dgamma")
    close(dbeta, np.array([1.0, 0.0, 0.0]), atol=1e-12, label="dbeta")

    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 7))
    gamma = rng.normal(size=7) + 1.5
    beta = rng.normal(size=7)
    dout = rng.normal(size=(5, 7))

    _, cache = layernorm_forward(x, gamma, beta)
    dx, dgamma, dbeta = layernorm_backward(dout, cache)
    shape(dx, x.shape, label="dx")
    shape(dgamma, gamma.shape, label="dgamma")
    shape(dbeta, beta.shape, label="dbeta")

    f_x = lambda x_: float(np.sum(dout * layernorm_forward(x_, gamma, beta)[0]))
    f_gamma = lambda g_: float(np.sum(dout * layernorm_forward(x, g_, beta)[0]))
    f_beta = lambda b_: float(np.sum(dout * layernorm_forward(x, gamma, b_)[0]))

    close(dx, finite_difference(f_x, x), atol=1e-4, rtol=1e-3, label="dx vs finite difference")
    close(dgamma, finite_difference(f_gamma, gamma), atol=1e-4, rtol=1e-3, label="dgamma vs finite difference")
    close(dbeta, finite_difference(f_beta, beta), atol=1e-4, rtol=1e-3, label="dbeta vs finite difference")


@check("dx sums to zero over each normalised group and is orthogonal to xhat")
def test_dx_properties(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    layernorm_forward = has(sub, "layernorm_forward")
    batchnorm_backward = has(sub, "batchnorm_backward")
    layernorm_backward = has(sub, "layernorm_backward")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 5))
    dout = rng.normal(size=(8, 5))
    gamma, beta = rng.normal(size=5) + 1.0, rng.normal(size=5)

    _, cache = batchnorm_forward(x, gamma, beta)
    dx, _, _ = batchnorm_backward(dout, cache)
    close(dx.sum(axis=0), np.zeros(5), atol=1e-9, label="batch norm: column sums of dx")

    _, cache = layernorm_forward(x, gamma, beta)
    dx, _, _ = layernorm_backward(dout, cache)
    close(dx.sum(axis=1), np.zeros(8), atol=1e-9, label="layer norm: row sums of dx")

    rng = np.random.default_rng(6)
    x = rng.normal(size=(6, 40))
    dout = rng.normal(size=(6, 40))
    _, cache = layernorm_forward(x, np.ones(40), np.zeros(40))
    dx, _, _ = layernorm_backward(dout, cache)
    inner = (dx * cache[0]).sum(axis=1)
    close(inner, np.zeros(6), atol=1e-3, label="row inner products of dx and xhat")


@check("layer norm backward is batch norm backward on the transpose")
def test_transpose_relation(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    layernorm_forward = has(sub, "layernorm_forward")
    batchnorm_backward = has(sub, "batchnorm_backward")
    layernorm_backward = has(sub, "layernorm_backward")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 9))
    dout = rng.normal(size=(5, 9))
    _, c_ln = layernorm_forward(x, np.ones(9), np.zeros(9))
    dx_ln, _, _ = layernorm_backward(dout, c_ln)
    _, c_bn = batchnorm_forward(x.T, np.ones(5), np.zeros(5))
    dx_bn, _, _ = batchnorm_backward(dout.T, c_bn)
    close(dx_ln, dx_bn.T, atol=1e-9, label="layer norm dx vs batch norm dx on the transpose")
