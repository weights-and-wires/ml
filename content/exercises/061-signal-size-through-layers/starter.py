"""Predict and measure how a signal's size changes through a stack of layers.

A linear layer multiplies the mean square of its signal by `n_in * scale**2`.
You write that factor, the factor after many layers, and a simulator that
measures the real size so you can compare it with the prediction.
"""

from __future__ import annotations

import numpy as np


def layer_gain(n_in: int, scale: float) -> float:
    """Return the factor one linear layer multiplies the mean square by.

    Args:
        n_in: Number of inputs to each output unit (the layer's fan-in).
        scale: Standard deviation of the weights.

    Returns:
        `n_in * scale**2`, as a float.
    """
    raise NotImplementedError("layer_gain")


def depth_gain(gain: float, depth: int) -> float:
    """Return the factor `depth` identical layers multiply the mean square by.

    Args:
        gain: The factor of a single layer.
        depth: The number of layers.

    Returns:
        `gain ** depth`, as a float.
    """
    raise NotImplementedError("depth_gain")


def forward_variance(
    depth: int,
    widths: list[int],
    scale: float,
    activation: str,
    num_trials: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Push random inputs through random layers and record the mean square.

    Create `rng = np.random.default_rng(seed)`. Draw the input `h` with shape
    `(num_trials, widths[0])` from a standard normal. Then, for each of `depth`
    layers, draw `W` of shape `(widths[l], widths[l + 1])` from a normal with
    mean 0 and standard deviation `scale`, and set `h = h @ W`. When
    `activation` is `"relu"`, replace negatives in `h` with 0 after the
    multiply. There is no bias, and the input is never passed through ReLU.

    Args:
        depth: Number of layers.
        widths: `depth + 1` layer widths, input first.
        scale: Standard deviation of every weight.
        activation: `"linear"` or `"relu"`.
        num_trials: Number of random input vectors.
        seed: Seed for the random generator.

    Returns:
        An array of length `depth + 1`: `np.mean(h ** 2)` for the input and
        after each layer, averaged over every trial and every unit.
    """
    raise NotImplementedError("forward_variance")


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
