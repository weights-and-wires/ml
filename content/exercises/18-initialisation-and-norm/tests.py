"""Checks for the initialisation and normalisation exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("xavier_scale returns sqrt(2 / (n_in + n_out))")
def test_xavier_scale(sub):
    xavier_scale = has(sub, "xavier_scale")
    almost(xavier_scale(100, 100), np.sqrt(2 / 200), tol=1e-10, label="square layer")
    almost(xavier_scale(50, 200), np.sqrt(2 / 250), tol=1e-10, label="rectangular layer")


@check("he_scale returns sqrt(2 / n_in)")
def test_he_scale(sub):
    he_scale = has(sub, "he_scale")
    almost(he_scale(100), np.sqrt(2 / 100), tol=1e-10, label="n_in=100")
    almost(he_scale(512), np.sqrt(2 / 512), tol=1e-10, label="n_in=512")


@check("forward_variance reproduces the one-layer linear recursion n * Var(W)")
def test_forward_variance_single_layer(sub):
    forward_variance = has(sub, "forward_variance")
    n, scale = 300, 0.05
    variances = forward_variance(1, [n, n], scale, "linear", num_trials=3000, seed=0)
    shape(variances, (2,), label="forward_variance output")
    almost(variances[0], 1.0, tol=0.1, label="input variance")
    predicted = n * scale**2
    almost(variances[1], predicted, tol=0.15 * predicted, label="variance after one linear layer")


@check("batchnorm_forward gives each feature zero mean and unit variance over the batch")
def test_batchnorm_forward(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(500, 6))
    gamma, beta = np.ones(6), np.zeros(6)
    out, _ = batchnorm_forward(x, gamma, beta)
    shape(out, x.shape, label="batchnorm_forward output")
    close(out.mean(axis=0), np.zeros(6), atol=1e-6, label="per-feature mean")
    close(out.var(axis=0), np.ones(6), atol=1e-3, label="per-feature variance")


@check("layernorm_forward gives each example zero mean and unit variance over its features")
def test_layernorm_forward(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    rng = np.random.default_rng(1)
    x = rng.normal(loc=-2.0, scale=4.0, size=(50, 200))
    gamma, beta = np.ones(200), np.zeros(200)
    out, _ = layernorm_forward(x, gamma, beta)
    shape(out, x.shape, label="layernorm_forward output")
    close(out.mean(axis=1), np.zeros(50), atol=1e-6, label="per-example mean")
    close(out.var(axis=1), np.ones(50), atol=1e-3, label="per-example variance")


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


@check("layernorm_backward matches finite differences for x, gamma, and beta")
def test_layernorm_backward(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    layernorm_backward = has(sub, "layernorm_backward")
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


@check("layer norm is invariant to shifting a whole example by a constant; batch norm is not")
def test_layernorm_invariance(sub):
    batchnorm_forward = has(sub, "batchnorm_forward")
    layernorm_forward = has(sub, "layernorm_forward")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 6))
    gamma, beta = np.ones(6), np.zeros(6)
    shifted = x.copy()
    shifted[0] += 37.5  # add the same constant to every feature of one example

    ln_before, _ = layernorm_forward(x, gamma, beta)
    ln_after, _ = layernorm_forward(shifted, gamma, beta)
    close(ln_after[0], ln_before[0], atol=1e-6, label="layer norm's row 0 after the shift")

    bn_before, _ = batchnorm_forward(x, gamma, beta)
    bn_after, _ = batchnorm_forward(shifted, gamma, beta)
    moved = float(np.abs(bn_after[0] - bn_before[0]).max())
    if moved < 1e-3:
        fail(
            "batch norm normalises across the batch, so shifting one example changes that "
            f"example's batch statistics and its output should move, but it moved only {moved:.3g}"
        )


@check("He initialisation keeps a 50-layer ReLU network's variance flat; Xavier decays it by one half per layer")
def test_he_vs_xavier_depth(sub):
    forward_variance = has(sub, "forward_variance")
    he_scale = has(sub, "he_scale")
    xavier_scale = has(sub, "xavier_scale")
    width, depth = 128, 50

    var_he = forward_variance(depth, [width] * (depth + 1), he_scale(width), "relu", num_trials=300, seed=0)
    var_xavier = forward_variance(
        depth, [width] * (depth + 1), xavier_scale(width, width), "relu", num_trials=300, seed=0
    )
    shape(var_he, (depth + 1,), label="forward_variance output length")

    he_factor = float((var_he[-1] / var_he[0]) ** (1.0 / depth))
    xavier_factor = float((var_xavier[-1] / var_xavier[0]) ** (1.0 / depth))

    almost(he_factor, 1.0, tol=0.15, label="He's effective per-layer variance factor")
    almost(xavier_factor, 0.5, tol=0.1, label="Xavier's effective per-layer variance factor")
