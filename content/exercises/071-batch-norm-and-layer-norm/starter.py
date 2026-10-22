"""Normalise a group of numbers, then a batch by column and by row.

Normalising rescales a group of numbers to mean 0 and variance 1. Batch norm
takes each column of an `(N, D)` array as a group, and layer norm takes each
row. Both then apply a learned scale `gamma` and shift `beta`, each of shape
`(D,)`.
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
        `1 / sqrt(var + eps)` with shape `(1, D)` and `axis` is 0. The next
        chapter's backward pass reads it.
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
