"""Convolution by hand: slide a filter, count the weights, count the output, count the view.

Fill in the nine functions below. The loop version of the convolution is slow on
purpose: later chapters compare faster versions against it.
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


def conv_output_shape(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """Return the output length along one axis.

    Args:
        size: Input length along this axis.
        kernel_size: Number of kernel taps along this axis.
        stride: Step between consecutive windows.
        padding: Zeros added to each side before sliding the window.
        dilation: Spacing between taps; 1 means adjacent taps.

    Returns:
        The number of window positions that fit. Leftover cells at the far end
        that no window reaches are dropped (round down).
    """
    raise NotImplementedError("conv_output_shape")


def receptive_field_per_layer(layers: list[tuple[int, int]]) -> list[int]:
    """Return the receptive field, in input pixels, after each layer.

    Args:
        layers: A list of (kernel_size, stride) pairs, from the layer that
            reads the input to the last layer.

    Returns:
        A list with one integer per layer: how many input pixels along one axis
        a single unit of that layer depends on. The last entry is the answer for
        the whole stack. Padding does not change the field.
    """
    raise NotImplementedError("receptive_field_per_layer")


def max_pool_forward(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    """Max pooling over square windows, separately for every channel.

    Args:
        x: Input, shape (N, C, H, W).
        pool_size: Side length of the square window.
        stride: Step between consecutive windows; may be smaller than
            pool_size, in which case windows overlap.

    Returns:
        Output, shape (N, C, H_out, W_out), where H_out and W_out come from
        `conv_output_shape` with padding 0 and dilation 1.
    """
    raise NotImplementedError("max_pool_forward")


def max_pool_backward(
    dout: np.ndarray, x: np.ndarray, pool_size: int, stride: int
) -> np.ndarray:
    """Route each max-pool gradient to the first maximum in its window.

    Args:
        dout: Upstream gradient, shape (N, C, H_out, W_out), the shape
            `max_pool_forward` returns for this `x`.
        x: The same input that was pooled, shape (N, C, H, W).
        pool_size: Side length of the square window.
        stride: Step between consecutive windows.

    Returns:
        Gradient with respect to `x`, shape (N, C, H, W). On a tie, the whole
        upstream value goes to the first maximum in row-major order (the index
        `np.argmax` returns on the flattened window) and the later ties get
        zero. Overlapping windows add. Cells that no window covers stay zero.
    """
    raise NotImplementedError("max_pool_backward")


def average_pool_forward(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    """Average pooling over square windows, separately for every channel.

    Args:
        x: Input, shape (N, C, H, W).
        pool_size: Side length of the square window.
        stride: Step between consecutive windows; may be smaller than
            pool_size, in which case windows overlap.

    Returns:
        Output, shape (N, C, H_out, W_out), where H_out and W_out come from
        `conv_output_shape` with padding 0 and dilation 1. Each value is the
        mean of its window.
    """
    raise NotImplementedError("average_pool_forward")


def average_pool_backward(
    dout: np.ndarray, x: np.ndarray, pool_size: int, stride: int
) -> np.ndarray:
    """Split each average-pool gradient evenly across its window.

    Args:
        dout: Upstream gradient, shape (N, C, H_out, W_out).
        x: The input that was pooled, shape (N, C, H, W). Only the shape is
            needed; leftover cells that no window covers stay zero.
        pool_size: Side length of the square window.
        stride: Step between consecutive windows.

    Returns:
        Gradient with respect to `x`, shape (N, C, H, W). Every cell of a
        window receives `dout / (pool_size * pool_size)`. Overlapping windows
        add.
    """
    raise NotImplementedError("average_pool_backward")


if __name__ == "__main__":
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

    stack = [(3, 2, 1), (3, 2, 1), (3, 1, 1)]
    size = 32
    for k, s, p in stack:
        size = conv_output_shape(size, k, s, p)
        print(f"after a {k}x{k} layer with stride {s} and padding {p}: {size} by {size}")
    print("receptive field after each layer:",
          receptive_field_per_layer([(k, s) for k, s, _ in stack]))

    grid = np.array([[1, 3, 2, 5], [4, 9, 1, 0], [6, 2, 7, 1], [1, 1, 3, 2]], dtype=float)
    print("2x2 max pool:")
    print(max_pool_forward(grid[None, None], 2, 2)[0, 0])
    dout = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    print("max-pool gradient:")
    print(max_pool_backward(dout, grid[None, None], 2, 2)[0, 0])
    print("2x2 average pool:")
    print(average_pool_forward(grid[None, None], 2, 2)[0, 0])
    print("average-pool gradient:")
    print(average_pool_backward(dout, grid[None, None], 2, 2)[0, 0])
