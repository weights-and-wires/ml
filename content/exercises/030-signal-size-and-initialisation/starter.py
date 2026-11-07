"""Predict a signal's size through a stack of layers, then set the weight scale.

A linear layer multiplies the mean square of its signal by the gain
`n_in * scale**2`, and `depth` layers multiply it by the gain `depth` times.
Xavier picks the scale so the average of the forward and backward gains is 1.
He doubles the weight variance so that a ReLU, which removes half of the mean
square, leaves a gain of 1. Fill in the six functions, in order.
"""

from __future__ import annotations

import numpy as np


def layer_gain(n_in: int, scale: float) -> float:
    """Return the factor one linear layer applies to the mean square of its signal.

    Args:
        n_in: Fan-in, the number of inputs to each output unit.
        scale: Standard deviation of the weights, not the variance.

    Returns:
        The gain `n_in * scale**2`, as a float.
    """
    raise NotImplementedError("layer_gain")


def depth_gain(gain: float, depth: int) -> float:
    """Return the factor that `depth` identical layers apply to the mean square.

    Args:
        gain: The gain of one layer.
        depth: Number of layers, 0 or more.

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
    """Simulate a random stack and return the signal's mean square at every layer.

    Draw the input `h` from a standard normal with shape `(num_trials, widths[0])`
    using `np.random.default_rng(seed)`. For each layer, draw weights of shape
    `(widths[l], widths[l + 1])` from a normal with standard deviation `scale`,
    and replace `h` with `h @ W`. When `activation` is `"relu"`, apply
    `np.maximum(h, 0.0)` after the multiply, never to the input.

    Args:
        depth: Number of layers.
        widths: Layer widths, `depth + 1` of them.
        scale: Standard deviation of the weights.
        activation: `"linear"` or `"relu"`. Any other value raises `ValueError`.
        num_trials: Number of random inputs.
        seed: Seed for the random generator.

    Returns:
        Array of shape `(depth + 1,)`: `np.mean(h ** 2)` for the input and after
        every layer.
    """
    raise NotImplementedError("forward_variance")


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
    width, depth = 128, 50
    for scale in (0.07, 1 / np.sqrt(width), 0.1):
        gain = layer_gain(width, scale)
        predicted = depth_gain(gain, depth)
        measured = forward_variance(depth, [width] * (depth + 1), scale, "linear")
        print(
            f"scale {scale:.4f}: gain {gain:.3f}, predicted size after {depth} layers "
            f"{predicted:.2e}, measured {measured[-1]:.2e}"
        )

    print(f"Xavier for 256 in, 64 out: {xavier_scale(256, 64):.4f}")
    rng = np.random.default_rng(0)
    print(f"ReLU keeps {relu_mean_square_ratio(rng.normal(size=100_000)):.3f} of the mean square")
    for name, scale in (("He", he_scale(width)), ("Xavier", xavier_scale(width, width))):
        sizes = forward_variance(depth, [width] * (depth + 1), scale, "relu")
        print(f"{name:6s}: mean square after {depth} ReLU layers = {sizes[-1]:.2e}")
