"""Turn a convolution into one matrix multiply, and send the gradient back.

You gather every window of the input into the columns of a matrix (im2col),
multiply by the kernels flattened into rows, and check the result against the
slow loop from an earlier chapter. Then you write the 1 by 1 convolution as a
plain matrix multiply, and build the backward pass: col2im, the scatter-add
that undoes im2col, a count of the windows that cover each pixel, and the
gradients dx, dw, and db.
"""

import numpy as np


# Written in "How big is the output and how far can a unit see".
def conv_output_shape(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """Return the output length along one axis (solved for you)."""
    reach = dilation * (kernel_size - 1) + 1
    return (size + 2 * padding - reach) // stride + 1


# Written in "Sliding one small filter over a picture".
def conv2d_naive(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Batched multi-channel convolution with plain loops (solved for you)."""
    n, c_in, h, w_dim = x.shape
    c_out, _, k, _ = w.shape
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    out = np.zeros((n, c_out, h_out, w_out))
    for img in range(n):
        for o in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    top, left = i * stride, j * stride
                    patch = xp[img, :, top : top + k, left : left + k]
                    out[img, o, i, j] = b[o] + np.sum(w[o] * patch)
    return out


def im2col(x: np.ndarray, kernel_size: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Lay every window of x out as a column.

    Args:
        x: Input, shape (N, C, H, W).
        kernel_size: Side length k of the square window.
        stride: Step between windows.
        padding: Zeros added to each spatial side before gathering.

    Returns:
        Array of shape (N, C * k * k, H_out * W_out), with H_out and W_out from
        `conv_output_shape`. Column l holds the window for output position
        (l // W_out, l % W_out), flattened in (channel, kernel row, kernel
        column) order, so that it lines up with `w.reshape(C_out, -1)`.
    """
    raise NotImplementedError("im2col")


def conv2d_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Batched 2-D convolution as one matrix multiply on the output of im2col.

    Args:
        x: Input, shape (N, C_in, H, W).
        w: Kernels, shape (C_out, C_in, k, k).
        b: Biases, shape (C_out,).
        stride: Stride along both spatial axes.
        padding: Zero padding on every side.

    Returns:
        Output, shape (N, C_out, H_out, W_out). No Python loop over output
        positions, channels, or images.
    """
    raise NotImplementedError("conv2d_forward")


def conv1x1_as_matmul(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A 1 by 1 convolution written as one matrix multiply over channels.

    Args:
        x: Input, shape (N, C_in, H, W).
        w: Kernels, shape (C_out, C_in, 1, 1).
        b: Biases, shape (C_out,).

    Returns:
        Output, shape (N, C_out, H, W). Do not call `im2col`: every window is a
        single pixel, so the columns are just the pixels with their channels.
    """
    raise NotImplementedError("conv1x1_as_matmul")


def col2im(
    dcol: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Scatter-add columns back into an image: the transpose of im2col.

    Where several windows cover the same pixel, their entries add up.

    Args:
        dcol: Array shaped like the output of `im2col`:
            (N, C * k * k, H_out * W_out).
        x_shape: The (N, C, H, W) shape of the original, unpadded image.
        kernel_size: Side length of the square window.
        stride: Stride used to build the columns.
        padding: Padding used to build the columns. The returned array has the
            padding cut off again.

    Returns:
        Array of shape x_shape.
    """
    raise NotImplementedError("col2im")


def window_coverage(
    x_shape: tuple[int, int, int, int], kernel_size: int, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Count how many windows cover each pixel of the unpadded image.

    Args:
        x_shape: The (N, C, H, W) shape of the input.
        kernel_size: Side length of the square window.
        stride: Stride of the convolution.
        padding: Padding of the convolution. Windows that overlap the padding
            still count only for the real pixels they cover.

    Returns:
        Integer-valued array of shape (H, W). Pixels no window reads count 0.
    """
    raise NotImplementedError("window_coverage")


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
        dY: Gradient of the loss with respect to the output,
            shape (N, C_out, H_out, W_out).
        x: The forward pass's input, shape (N, C_in, H, W).
        w: The forward pass's kernels, shape (C_out, C_in, k, k).
        b: The forward pass's biases, shape (C_out,).
        stride: Stride used in the forward pass.
        padding: Padding used in the forward pass.

    Returns:
        (dx, dw, db), with the same shapes as (x, w, b). Weight and bias
        gradients add up over every image in the batch and every position.
    """
    raise NotImplementedError("conv2d_backward")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 9, 9))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.2
    b = rng.normal(size=4) * 0.1

    col = im2col(x, 3, stride=1, padding=1)
    print("col shape:", col.shape, "from input shape", x.shape)
    print("entries in col divided by entries in x:", round(col.size / x.size, 2))

    fast = conv2d_forward(x, w, b, stride=2, padding=1)
    slow = conv2d_naive(x, w, b, stride=2, padding=1)
    print("output shape:", fast.shape)
    print("largest difference from the loop version:", np.abs(fast - slow).max())

    w1 = rng.normal(size=(5, 3, 1, 1))
    mix = conv1x1_as_matmul(x, w1, np.zeros(5))
    print("1x1 output shape:", mix.shape)

    # A 3 by 3 image, a 2 by 2 kernel, and dY all ones.
    x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    b = np.zeros(1)
    dY = np.ones((1, 1, 2, 2))
    dx, dw, db = conv2d_backward(dY, x, w, b)
    print("coverage counts:")
    print(window_coverage(x.shape, 2))
    print("dw:", dw[0, 0].tolist(), " db:", db.tolist())
    print("dx:")
    print(dx[0, 0])
