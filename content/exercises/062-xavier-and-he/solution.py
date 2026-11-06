"""Xavier and He weight scales, and what a ReLU does to a signal's size."""

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
    return float(np.sqrt(2.0 / (n_in + n_out)))


def relu_mean_square_ratio(preact: np.ndarray) -> float:
    after = np.maximum(preact, 0.0)
    return float(np.mean(after**2) / np.mean(preact**2))


def he_scale(n_in: int) -> float:
    return float(np.sqrt(2.0 / n_in))


if __name__ == "__main__":
    print(f"Xavier for 256 in, 64 out: {xavier_scale(256, 64):.4f}")
    rng = np.random.default_rng(0)
    print(f"ReLU keeps {relu_mean_square_ratio(rng.normal(size=100_000)):.3f} of the mean square")

    width, depth = 128, 50
    for name, scale in (("He", he_scale(width)), ("Xavier", xavier_scale(width, width))):
        sizes = forward_variance(depth, [width] * (depth + 1), scale, "relu")
        print(f"{name:6s}: mean square after {depth} ReLU layers = {sizes[-1]:.2e}")
