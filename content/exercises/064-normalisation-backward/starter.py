"""Backward passes for batch norm and layer norm.

Every input in a group changes every output of that group through the shared
mean and variance, so the gradient of one input has three routes: directly,
through the mean, and through the variance. They collapse into one compact
formula that is the same for both layers apart from the axis it sums over.
"""

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
    """Return the gradients of the learned scale and shift.

    Args:
        dout: Gradient of the loss with respect to the layer's output, shape `(N, D)`.
        xhat: The normalised values from the forward pass, shape `(N, D)`.

    Returns:
        A pair `(dgamma, dbeta)`, each of shape `(D,)`. Both sum over axis 0,
        because every example in a column uses the same `gamma` and `beta`,
        for batch norm and for layer norm alike.
    """
    raise NotImplementedError("affine_param_grads")


def batchnorm_backward(dout: np.ndarray, cache):
    """Send the gradient back through batch norm.

    Args:
        dout: Gradient of the loss with respect to the output, shape `(N, D)`.
        cache: The tuple `(xhat, istd, gamma, axis)` from `batchnorm_forward`.

    Returns:
        A tuple `(dx, dgamma, dbeta)` with shapes `(N, D)`, `(D,)`, and `(D,)`.
        With `dxhat = dout * gamma` and `M` the size of the group,
        `dx = (istd / M) * (M * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))`,
        where each sum runs over the normalised axis with `keepdims=True`.
    """
    raise NotImplementedError("batchnorm_backward")


def layernorm_backward(dout: np.ndarray, cache):
    """Send the gradient back through layer norm.

    Args:
        dout: Gradient of the loss with respect to the output, shape `(N, D)`.
        cache: The tuple `(xhat, istd, gamma, axis)` from `layernorm_forward`.

    Returns:
        A tuple `(dx, dgamma, dbeta)` with shapes `(N, D)`, `(D,)`, and `(D,)`.
        The `dx` formula is the one in `batchnorm_backward`, summed over the
        features of each row, so `M` is `D`.
    """
    raise NotImplementedError("layernorm_backward")


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
