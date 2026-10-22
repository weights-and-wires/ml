"""Slide a filter over a picture, one position at a time.

You write the one-channel sliding filter, then a batched multi-channel version
built from plain loops, then a small function that counts the weights of three
kinds of layer. The loop version is slow on purpose: later chapters compare
faster versions against it.
"""

import numpy as np


def conv2d_single(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Slide one k by k kernel over one single-channel image.

    The kernel is not flipped: output (i, j) is the bias plus the sum of
    w[a, c] * x[i + a, j + c] over the window whose top-left corner is (i, j).

    Args:
        x: Image, shape (H, W).
        w: Kernel, shape (k, k).
        b: Bias, one number.

    Returns:
        Output grid, shape (H - k + 1, W - k + 1): stride 1, no padding.
    """
    raise NotImplementedError("conv2d_single")


def conv2d_naive(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Batched multi-channel convolution, written with plain loops.

    Args:
        x: Input, shape (N, C_in, H, W).
        w: Kernels, shape (C_out, C_in, k, k). Each output channel has one
            kernel that reads every input channel at once.
        b: Biases, shape (C_out,).
        stride: Step between windows, along both spatial axes.
        padding: Zeros added to every side of the image before sliding.

    Returns:
        Output, shape (N, C_out, H_out, W_out), where
        H_out = (H + 2 * padding - k) // stride + 1, and W_out likewise.
    """
    raise NotImplementedError("conv2d_naive")


def param_counts(n_in: int, n_out: int, c_in: int, c_out: int, k: int) -> dict:
    """Count the learnable numbers (weights plus biases) in three layers.

    Args:
        n_in: Inputs to the dense layer (for a photo, H * W * channels).
        n_out: Units in the dense layer.
        c_in: Input channels of the convolution layers.
        c_out: Output channels (filters) of the convolution layers.
        k: Side length of the square kernel.

    Returns:
        A dict with three integer entries:
        "dense": a fully connected layer from n_in numbers to n_out units.
        "conv": a convolution layer with c_out filters of shape (c_in, k, k).
        "separable": a depthwise separable layer: one k by k filter per input
            channel, then a 1 by 1 convolution that mixes channels, with one
            bias per output channel.
    """
    raise NotImplementedError("param_counts")


if __name__ == "__main__":
    # The chapter's picture: a bright region on the right, a dark one on the left.
    x = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)
    edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    print("vertical-edge response:")
    print(conv2d_single(x, edge, 0.0))

    rng = np.random.default_rng(0)
    imgs = rng.normal(size=(2, 3, 8, 8))
    kernels = rng.normal(size=(4, 3, 3, 3)) * 0.2
    out = conv2d_naive(imgs, kernels, np.zeros(4), stride=2, padding=1)
    print("batched output shape:", out.shape)

    counts = param_counts(224 * 224 * 3, 1000, 3, 64, 3)
    print("photo layer, dense vs conv:", counts["dense"], "vs", counts["conv"])
    counts = param_counts(0, 0, 64, 64, 3)
    print("64 to 64 channels, conv vs separable:", counts["conv"], "vs", counts["separable"])
