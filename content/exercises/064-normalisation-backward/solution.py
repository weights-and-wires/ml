"""Backward passes for batch norm and layer norm."""

from __future__ import annotations

import numpy as np


# Written in the chapter "Normalising activations: batch norm versus layer norm".
def _normalise(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float, axis: int):
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    istd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * istd
    out = gamma * xhat + beta
    return out, (xhat, istd, gamma, axis)


# Written in the chapter "Normalising activations: batch norm versus layer norm".
def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    return _normalise(x, gamma, beta, eps, axis=0)


# Written in the chapter "Normalising activations: batch norm versus layer norm".
def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    return _normalise(x, gamma, beta, eps, axis=1)


def affine_param_grads(dout: np.ndarray, xhat: np.ndarray):
    dgamma = (dout * xhat).sum(axis=0)
    dbeta = dout.sum(axis=0)
    return dgamma, dbeta


def _normalise_backward(dout: np.ndarray, cache):
    xhat, istd, gamma, axis = cache
    m = dout.shape[axis]
    dgamma, dbeta = affine_param_grads(dout, xhat)
    dxhat = dout * gamma
    dx = (istd / m) * (
        m * dxhat
        - dxhat.sum(axis=axis, keepdims=True)
        - xhat * (dxhat * xhat).sum(axis=axis, keepdims=True)
    )
    return dx, dgamma, dbeta


def batchnorm_backward(dout: np.ndarray, cache):
    return _normalise_backward(dout, cache)


def layernorm_backward(dout: np.ndarray, cache):
    return _normalise_backward(dout, cache)


if __name__ == "__main__":
    x = np.array([[1.0, 2.0, 3.0]])
    gamma, beta = np.ones(3), np.zeros(3)
    out, cache = layernorm_forward(x, gamma, beta)
    dout = np.array([[1.0, 0.0, 0.0]])
    dx, dgamma, dbeta = layernorm_backward(dout, cache)
    print("dx:", np.round(dx, 4), "sums to", round(float(dx.sum()), 8))
    print("dgamma:", np.round(dgamma, 4), "dbeta:", dbeta)

    h = 1e-6
    def f(v):
        return float(np.sum(dout * layernorm_forward(np.array([[v, 2.0, 3.0]]), gamma, beta)[0]))
    print("finite difference for x[0]:", round((f(1.0 + h) - f(1.0 - h)) / (2 * h), 5))
