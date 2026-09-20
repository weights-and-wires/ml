"""Convolution as a matrix multiply.

Implement a batched 2-D convolution by gathering patches with `im2col` and
reducing the forward and backward pass to matrix products, plus the two
formulas that describe a conv stack from the outside: the output-size formula
and the receptive-field recursion.

Every tensor is a plain `numpy.ndarray`. The kernel is always square: a kernel
of "size k" has shape (k, k) along its last two axes.
"""

import numpy as np


def conv_output_shape(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """Return the output length along one spatial axis.

    Args:
        size: Input length along this axis.
        kernel_size: Number of kernel taps along this axis.
        stride: Step between consecutive windows.
        padding: Zeros added to each side before sliding the window.
        dilation: Spacing between taps; 1 means adjacent taps.

    Returns:
        The number of positions the (possibly dilated) kernel fits.
    """
    raise NotImplementedError("conv_output_shape")


def im2col(x: np.ndarray, kernel_size: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Gather every convolution patch of x into the columns of a matrix.

    Args:
        x: Input, shape (N, C, H, W).
        kernel_size: Side length of the square kernel.
        stride: Step between consecutive windows.
        padding: Zeros added to each spatial side before gathering.

    Returns:
        Array of shape (N, C * kernel_size * kernel_size, H_out * W_out), where
        H_out and W_out come from `conv_output_shape`. Column l holds the patch
        for output position (l // W_out, l % W_out), flattened in
        (channel, kernel_row, kernel_col) order to match a kernel reshaped with
        `w.reshape(C_out, -1)`.
    """
    raise NotImplementedError("im2col")


def col2im(
    dcol: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Undo im2col: scatter-add columns back into an image of shape x_shape.

    This is the adjoint of `im2col`, not its inverse: positions covered by more
    than one patch accumulate the sum of every patch's contribution, which is
    exactly the operation the chain rule needs to turn a gradient with respect
    to the columns back into a gradient with respect to the image.

    Args:
        dcol: Array shaped like the output of `im2col`.
        x_shape: The (N, C, H, W) shape of the original, unpadded image.
        kernel_size: Side length of the square kernel used to build dcol.
        stride: Stride used to build dcol.
        padding: Padding used to build dcol.

    Returns:
        Array of shape x_shape.
    """
    raise NotImplementedError("col2im")


def conv2d_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Batched 2-D convolution, computed through im2col as a matrix multiply.

    Args:
        x: Input, shape (N, C_in, H, W).
        w: Kernel, shape (C_out, C_in, k, k).
        b: Bias, shape (C_out,).
        stride: Stride applied to both spatial axes.
        padding: Zero padding applied to both spatial axes.

    Returns:
        Output, shape (N, C_out, H_out, W_out).
    """
    raise NotImplementedError("conv2d_forward")


def conv2d_backward(
    dY: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass of conv2d_forward.

    Args:
        dY: Upstream gradient, shape (N, C_out, H_out, W_out).
        x: The forward pass's input, shape (N, C_in, H, W).
        w: The forward pass's kernel, shape (C_out, C_in, k, k).
        b: The forward pass's bias, shape (C_out,).
        stride: Stride used in the forward pass.
        padding: Padding used in the forward pass.

    Returns:
        A tuple (dx, dw, db) with the same shapes as (x, w, b).
    """
    raise NotImplementedError("conv2d_backward")


def max_pool_forward(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    """Max pooling over non-overlapping or overlapping square windows.

    Args:
        x: Input, shape (N, C, H, W).
        pool_size: Side length of the square pooling window.
        stride: Step between consecutive windows.

    Returns:
        Output, shape (N, C, H_out, W_out), with H_out and W_out from
        `conv_output_shape` at padding 0 and dilation 1.
    """
    raise NotImplementedError("max_pool_forward")


def receptive_field(layers: list[tuple[int, int]]) -> int:
    """Return the receptive field, in input pixels, at the end of a conv stack.

    Args:
        layers: A list of (kernel_size, stride) pairs, one per layer, in order
            from the first layer applied to the input to the last.

    Returns:
        The receptive field size of a single output unit of the last layer,
        via the recursion r_l = r_{l-1} + (k_l - 1) * prod_{i<l} s_i, r_0 = 1.
    """
    raise NotImplementedError("receptive_field")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 9, 9))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.1
    b = rng.normal(size=(4,)) * 0.1

    y = conv2d_forward(x, w, b, stride=2, padding=1)
    print(f"conv output shape: {y.shape}")

    dY = rng.normal(size=y.shape)
    dx, dw, db = conv2d_backward(dY, x, w, b, stride=2, padding=1)
    print(f"dx shape: {dx.shape}, dw shape: {dw.shape}, db shape: {db.shape}")

    pooled = max_pool_forward(x, pool_size=3, stride=3)
    print(f"pooled shape: {pooled.shape}")

    stack = [(3, 2), (3, 2), (3, 1)]
    print(f"receptive field of {stack}: {receptive_field(stack)}")
