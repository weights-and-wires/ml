"""Output sizes, receptive fields, and max pooling.

Three small tools for reasoning about a stack of convolution layers from the
outside: how long is each output axis, how much of the input can one unit see,
and what does a max-pooling layer keep.
"""

import numpy as np


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


if __name__ == "__main__":
    # Three layers on a 32 by 32 image: (kernel, stride, padding).
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
