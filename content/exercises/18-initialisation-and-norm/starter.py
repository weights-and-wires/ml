"""Initialisation scales, a signal-propagation simulator, and two norm layers.

Fill in the seven functions below. `batchnorm_forward` and `layernorm_forward`
each return `(out, cache)`, where `cache` holds whatever their matching
backward function needs; the two are the same normalisation formula applied
along different axes.
"""

import numpy as np


def xavier_scale(n_in: int, n_out: int) -> float:
    """Return the Xavier (Glorot) standard deviation for a linear layer.

    Args:
        n_in: Fan-in of the layer.
        n_out: Fan-out of the layer.

    Returns:
        The standard deviation `sqrt(2 / (n_in + n_out))` to draw weights from.
    """
    raise NotImplementedError("xavier_scale")


def he_scale(n_in: int) -> float:
    """Return the He standard deviation for a linear layer feeding a ReLU.

    Args:
        n_in: Fan-in of the layer.

    Returns:
        The standard deviation `sqrt(2 / n_in)` to draw weights from.
    """
    raise NotImplementedError("he_scale")


def forward_variance(
    depth: int,
    widths: list[int],
    scale: float,
    activation: str,
    num_trials: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Simulate a random forward stack and measure variance at every layer.

    Build `depth` weight matrices, layer `l` of shape `(widths[l], widths[l+1])`
    with i.i.d. `N(0, scale**2)` entries, and no bias. For `num_trials`
    independent draws of an input `x ~ N(0, I_{widths[0]})`, propagate

        h_0 = x
        h_l = activation(h_{l-1} @ W_l),  l = 1, ..., depth

    where `activation` is `"linear"` (identity) or `"relu"`. Track
    `mean(h_l ** 2)`, pooled over both the trials and the units of that layer:
    it equals the variance for the input and for every pre-activation, since
    those are zero-mean, and it is the mean-square activation that governs
    the next layer's variance even where `h_l` itself is not centred, as
    after a ReLU.

    Args:
        depth: Number of weight matrices to apply.
        widths: Layer widths, length `depth + 1`; `widths[0]` is the input
            dimension.
        scale: Standard deviation to draw every weight matrix's entries from.
        activation: `"linear"` or `"relu"`.
        num_trials: Number of independent input draws to average over.
        seed: Seed for the random number generator.

    Returns:
        An array of shape `(depth + 1,)`, the empirical variance of `h_l` for
        `l = 0, ..., depth`.
    """
    raise NotImplementedError("forward_variance")


def batchnorm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
):
    """Normalise `x` to zero mean and unit variance per feature, over the batch.

    Args:
        x: Activations, shape `(N, D)`.
        gamma: Per-feature scale, shape `(D,)`.
        beta: Per-feature shift, shape `(D,)`.
        eps: Added to the variance before the square root.

    Returns:
        A tuple `(out, cache)`. `out` has shape `(N, D)`. `cache` is whatever
        `batchnorm_backward` needs.
    """
    raise NotImplementedError("batchnorm_forward")


def batchnorm_backward(dout: np.ndarray, cache):
    """Backpropagate through `batchnorm_forward`.

    Args:
        dout: Upstream gradient, shape `(N, D)`, matching the forward output.
        cache: The cache `batchnorm_forward` returned.

    Returns:
        A tuple `(dx, dgamma, dbeta)` with shapes `(N, D)`, `(D,)`, `(D,)`.
    """
    raise NotImplementedError("batchnorm_backward")


def layernorm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
):
    """Normalise `x` to zero mean and unit variance per example, over features.

    Args:
        x: Activations, shape `(N, D)`.
        gamma: Per-feature scale, shape `(D,)`.
        beta: Per-feature shift, shape `(D,)`.
        eps: Added to the variance before the square root.

    Returns:
        A tuple `(out, cache)`. `out` has shape `(N, D)`. `cache` is whatever
        `layernorm_backward` needs.
    """
    raise NotImplementedError("layernorm_forward")


def layernorm_backward(dout: np.ndarray, cache):
    """Backpropagate through `layernorm_forward`.

    Args:
        dout: Upstream gradient, shape `(N, D)`, matching the forward output.
        cache: The cache `layernorm_forward` returned.

    Returns:
        A tuple `(dx, dgamma, dbeta)` with shapes `(N, D)`, `(D,)`, `(D,)`.
    """
    raise NotImplementedError("layernorm_backward")


if __name__ == "__main__":
    width = 128
    depth = 50

    he = he_scale(width)
    xavier = xavier_scale(width, width)
    var_he = forward_variance(depth, [width] * (depth + 1), he, "relu")
    var_xavier = forward_variance(depth, [width] * (depth + 1), xavier, "relu")
    print(f"He:     variance at layer 0 = {var_he[0]:.3f}, at layer {depth} = {var_he[-1]:.3e}")
    print(f"Xavier: variance at layer 0 = {var_xavier[0]:.3f}, at layer {depth} = {var_xavier[-1]:.3e}")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 8))
    gamma, beta = np.ones(8), np.zeros(8)

    bn_out, _ = batchnorm_forward(x, gamma, beta)
    ln_out, _ = layernorm_forward(x, gamma, beta)
    print(f"batchnorm per-feature mean (should be ~0): {bn_out.mean(axis=0)[:3]}")
    print(f"layernorm per-example mean (should be ~0): {ln_out.mean(axis=1)[:3]}")
