"""Reference solution: send the gradient back through a convolution."""

import numpy as np


# Written in "How big is the output and how far can a unit see".
def conv_output_shape(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """Return the output length along one axis (solved for you)."""
    reach = dilation * (kernel_size - 1) + 1
    return (size + 2 * padding - reach) // stride + 1


# Written in "Turning a convolution into one matrix multiply".
def im2col(x: np.ndarray, kernel_size: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Lay every window of x out as a column (solved for you).

    Returns an array of shape (N, C * k * k, H_out * W_out).
    """
    n, c, h, w_dim = x.shape
    k = kernel_size
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    col = np.zeros((n, c, k, k, h_out, w_out), dtype=x.dtype)
    for dh in range(k):
        for dw in range(k):
            col[:, :, dh, dw] = x_padded[
                :, :, dh : dh + stride * h_out : stride, dw : dw + stride * w_out : stride
            ]
    return col.reshape(n, c * k * k, h_out * w_out)


# Written in "Turning a convolution into one matrix multiply".
def conv2d_forward(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Convolution as one matrix multiply (solved for you)."""
    n, _, h, w_dim = x.shape
    c_out, _, k, _ = w.shape
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)
    col = im2col(x, k, stride, padding)
    out = np.einsum("oc,ncl->nol", w.reshape(c_out, -1), col) + b[None, :, None]
    return out.reshape(n, c_out, h_out, w_out)


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
    n, c, h, w_dim = x_shape
    k = kernel_size
    h_out = conv_output_shape(h, k, stride, padding)
    w_out = conv_output_shape(w_dim, k, stride, padding)

    dcol_r = dcol.reshape(n, c, k, k, h_out, w_out)
    canvas = np.zeros((n, c, h + 2 * padding, w_dim + 2 * padding), dtype=dcol.dtype)
    for dh in range(k):
        h_stop = dh + stride * h_out
        for dw in range(k):
            w_stop = dw + stride * w_out
            canvas[:, :, dh:h_stop:stride, dw:w_stop:stride] += dcol_r[:, :, dh, dw]

    if padding == 0:
        return canvas
    return canvas[:, :, padding:-padding, padding:-padding]


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
    _, _, h, w_dim = x_shape
    h_out = conv_output_shape(h, kernel_size, stride, padding)
    w_out = conv_output_shape(w_dim, kernel_size, stride, padding)
    ones = np.ones((1, kernel_size * kernel_size, h_out * w_out))
    return col2im(ones, (1, 1, h, w_dim), kernel_size, stride, padding)[0, 0]


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
    n = x.shape[0]
    c_out, _, k, _ = w.shape

    col = im2col(x, k, stride, padding)  # (N, C_in * k * k, L)
    w_mat = w.reshape(c_out, -1)  # (C_out, C_in * k * k)
    dY_mat = dY.reshape(n, c_out, -1)  # (N, C_out, L)

    db = dY_mat.sum(axis=(0, 2))
    dw_mat = np.einsum("nol,ncl->oc", dY_mat, col)
    dcol = np.einsum("oc,nol->ncl", w_mat, dY_mat)
    dx = col2im(dcol, x.shape, k, stride, padding)
    return dx, dw_mat.reshape(w.shape), db


if __name__ == "__main__":
    # The chapter's hand example: a 3 by 3 image, a 2 by 2 kernel, dY all ones.
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
