"""Normalise a group of numbers, then a batch by column and by row."""

from __future__ import annotations

import numpy as np


def normalise_group(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return (x - x.mean()) / np.sqrt(x.var() + eps)


def _normalise(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float, axis: int):
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    istd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * istd
    out = gamma * xhat + beta
    return out, (xhat, istd, gamma, axis)


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    return _normalise(x, gamma, beta, eps, axis=0)


def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    return _normalise(x, gamma, beta, eps, axis=1)


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
