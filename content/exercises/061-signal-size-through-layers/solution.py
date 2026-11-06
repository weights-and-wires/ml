"""Predict and measure how a signal's size changes through a stack of layers."""

from __future__ import annotations

import numpy as np


def layer_gain(n_in: int, scale: float) -> float:
    return float(n_in * scale**2)


def depth_gain(gain: float, depth: int) -> float:
    return float(gain**depth)


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


if __name__ == "__main__":
    width, depth = 128, 50
    for scale in (0.07, 1 / np.sqrt(width), 0.1):
        gain = layer_gain(width, scale)
        predicted = depth_gain(gain, depth)
        measured = forward_variance(depth, [width] * (depth + 1), scale, "linear")
        print(
            f"scale {scale:.4f}: gain {gain:.3f}, predicted size after {depth} layers "
            f"{predicted:.2e}, measured {measured[-1]:.2e}"
        )
