"""Checks for the normalisation backward exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


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


@check("layernorm_backward reproduces the hand example for the row [1, 2, 3]")
def test_hand_example(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    layernorm_backward = has(sub, "layernorm_backward")
    x = np.array([[1.0, 2.0, 3.0]])
    _, cache = layernorm_forward(x, np.ones(3), np.zeros(3))
    dx, dgamma, dbeta = layernorm_backward(np.array([[1.0, 0.0, 0.0]]), cache)
    close(dx, np.array([[0.2041, -0.4082, 0.2041]]), atol=1e-3, label="dx for dout = [1, 0, 0]")
    close(dgamma, np.array([-1.2247, 0.0, 0.0]), atol=1e-3, label="dgamma")
    close(dbeta, np.array([1.0, 0.0, 0.0]), atol=1e-12, label="dbeta")


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


@check("dx sums to zero over each normalised group")
def test_dx_sums_to_zero(sub):
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


@check("dx is orthogonal to xhat over each group, because rescaling changes nothing")
def test_dx_orthogonal_to_xhat(sub):
    layernorm_forward = has(sub, "layernorm_forward")
    layernorm_backward = has(sub, "layernorm_backward")
    rng = np.random.default_rng(6)
    x = rng.normal(size=(6, 40))
    dout = rng.normal(size=(6, 40))
    gamma, beta = np.ones(40), np.zeros(40)
    _, cache = layernorm_forward(x, gamma, beta)
    xhat = cache[0]
    dx, _, _ = layernorm_backward(dout, cache)
    inner = (dx * xhat).sum(axis=1)
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
