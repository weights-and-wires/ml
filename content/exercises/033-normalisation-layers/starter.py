"""Batch norm and layer norm, forward and backward.

Normalising rescales a group of numbers to mean 0 and variance 1. Batch norm
takes each column of an `(N, D)` array as a group, and layer norm takes each
row. Both then apply a learned scale `gamma` and shift `beta`, each of shape
`(D,)`. Every input in a group changes every output of that group through the
shared mean and variance, so the backward pass has three routes that collapse
into one compact formula. Fill in the six functions, in order. `_normalise`
and `_normalise_backward` are helpers you can write once and share.
"""

from __future__ import annotations

import numpy as np


def normalise_group(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Rescale one group of numbers to mean 0 and variance about 1.

    Args:
        x: Array of any shape. All of its entries form one group.
        eps: Small constant added to the variance before the square root.

    Returns:
        `(x - mean) / sqrt(var + eps)`, the same shape as `x`. The variance
        divides by the number of entries, as `np.var` does.
    """
    raise NotImplementedError("normalise_group")


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """Batch norm: normalise each feature (column) over the batch.

    Args:
        x: Array of shape `(N, D)`, N examples with D features.
        gamma: Learned scale, shape `(D,)`.
        beta: Learned shift, shape `(D,)`.
        eps: Small constant added to the variance.

    Returns:
        A pair `(out, cache)`. `out` has shape `(N, D)` and equals
        `gamma * xhat + beta`, where `xhat` is `x` normalised over axis 0.
        `cache` is the tuple `(xhat, istd, gamma, axis)`, where `istd` is
        `1 / sqrt(var + eps)` with shape `(1, D)` and `axis` is 0.
    """
    raise NotImplementedError("batchnorm_forward")


def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """Layer norm: normalise each example (row) over its own features.

    Args:
        x: Array of shape `(N, D)`.
        gamma: Learned scale, shape `(D,)`.
        beta: Learned shift, shape `(D,)`.
        eps: Small constant added to the variance.

    Returns:
        A pair `(out, cache)`. `out` has shape `(N, D)`. `cache` is the tuple
        `(xhat, istd, gamma, axis)`, where `istd` has shape `(N, 1)` and
        `axis` is 1.
    """
    raise NotImplementedError("layernorm_forward")


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
    row = np.array([1.0, 2.0, 3.0])
    print("normalise_group([1, 2, 3]):", np.round(normalise_group(row), 4))
    print("normalise_group([5, 5, 5]):", normalise_group(np.array([5.0, 5.0, 5.0])))

    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(6, 4))
    gamma, beta = np.ones(4), np.zeros(4)
    bn, _ = batchnorm_forward(x, gamma, beta)
    ln, _ = layernorm_forward(x, gamma, beta)
    print("batch norm column means:", np.round(bn.mean(axis=0), 6))
    print("layer norm row means:   ", np.round(ln.mean(axis=1), 6))
    one, _ = batchnorm_forward(x[:1], gamma, np.full(4, 0.5))
    print("batch norm of a single example equals beta:", one)

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
