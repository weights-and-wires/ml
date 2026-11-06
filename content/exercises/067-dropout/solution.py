"""Reference solution for the dropout exercise."""

import numpy as np


def dropout_mask(shape, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(shape) < 1.0 - p).astype(float)


def dropout_forward(x: np.ndarray, p: float, rng: np.random.Generator, training: bool):
    if not training:
        return x, np.ones_like(x)
    mask = dropout_mask(x.shape, p, rng)
    out = x * mask / (1.0 - p)
    return out, mask


def dropout_backward(dout: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    return dout * mask / (1.0 - p)


if __name__ == "__main__":
    x = np.array([2.0, 4.0, 6.0, 8.0])
    rng = np.random.default_rng(0)

    out, mask = dropout_forward(x, 0.5, rng, training=True)
    print("mask:", mask, "out:", out)

    sums = np.array([dropout_forward(x, 0.5, rng, training=True)[0].sum() for _ in range(2000)])
    print(f"mean layer sum over 2000 masks: {sums.mean():.2f} (test time gives {x.sum():.0f})")

    fixed = np.array([1.0, 0.0, 1.0, 1.0])
    print("backward for dout = ones:", dropout_backward(np.ones(4), fixed, 0.5))
