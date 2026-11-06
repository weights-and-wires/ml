"""Xavier and He weight scales, and what a ReLU does to a signal's size.

A linear layer's gain is `n_in * scale**2`. Xavier picks the scale so the average
of the forward and backward gains is 1. He doubles the weight variance so that a
ReLU, which removes half of the mean square, leaves a gain of 1.
"""

from __future__ import annotations

import numpy as np


# Written in the chapter "Why a deep network's signal shrinks or explodes".
def layer_gain(n_in: int, scale: float) -> float:
    return float(n_in * scale**2)


# Written in the chapter "Why a deep network's signal shrinks or explodes".
def forward_variance(
    depth: int,
    widths: list[int],
    scale: float,
    activation: str,
    num_trials: int = 300,
    seed: int = 0,
) -> np.ndarray:
    if activation not in ("linear", "relu"):
        raise ValueError(f"unknown activation {activation!r}")
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(num_trials, widths[0]))
    sizes = [float(np.mean(h**2))]
    for l in range(depth):
        w = rng.normal(scale=scale, size=(widths[l], widths[l + 1]))
        h = h @ w
        if activation == "relu":
            h = np.maximum(h, 0.0)
        sizes.append(float(np.mean(h**2)))
    return np.array(sizes)


def xavier_scale(n_in: int, n_out: int) -> float:
    """Return the Xavier (Glorot) standard deviation for a linear layer.

    Args:
        n_in: Fan-in, the number of inputs to each output unit.
        n_out: Fan-out, the number of output units each input feeds.

    Returns:
        The standard deviation `sqrt(2 / (n_in + n_out))` to draw weights from.
        It is a standard deviation, not a variance.
    """
    raise NotImplementedError("xavier_scale")


def relu_mean_square_ratio(preact: np.ndarray) -> float:
    """Return the fraction of the mean square that survives a ReLU.

    Args:
        preact: Array of any shape holding the numbers before the ReLU.

    Returns:
        `mean(relu(preact) ** 2) / mean(preact ** 2)`, as a float. For numbers
        spread symmetrically around 0 it is close to 0.5.
    """
    raise NotImplementedError("relu_mean_square_ratio")


def he_scale(n_in: int) -> float:
    """Return the He standard deviation for a linear layer followed by a ReLU.

    Args:
        n_in: Fan-in, the number of inputs to each output unit.

    Returns:
        The standard deviation `sqrt(2 / n_in)` to draw weights from.
    """
    raise NotImplementedError("he_scale")


if __name__ == "__main__":
    print(f"Xavier for 256 in, 64 out: {xavier_scale(256, 64):.4f}")
    rng = np.random.default_rng(0)
    print(f"ReLU keeps {relu_mean_square_ratio(rng.normal(size=100_000)):.3f} of the mean square")

    width, depth = 128, 50
    for name, scale in (("He", he_scale(width)), ("Xavier", xavier_scale(width, width))):
        sizes = forward_variance(depth, [width] * (depth + 1), scale, "relu")
        print(f"{name:6s}: mean square after {depth} ReLU layers = {sizes[-1]:.2e}")
