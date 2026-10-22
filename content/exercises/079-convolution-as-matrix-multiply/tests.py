"""Checks for the convolution-as-matrix-multiply exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def reference_conv(x, w, b, stride, padding):
    """An independent reference built from numpy's sliding windows."""
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, w.shape[2:], axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    return np.einsum("nchwab,ocab->nohw", windows, w) + b[None, :, None, None]


@check("im2col has the right shape and its first column is the top-left window")
def test_im2col_shape(sub):
    got = has(sub, "im2col")
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    col = got(x, 2, 1, 0)
    shape(col, (1, 4, 9), label="im2col output")
    close(col[0, :, 0], np.array([0.0, 1.0, 4.0, 5.0]), atol=1e-12, label="first column")
    close(col[0, :, 8], np.array([10.0, 11.0, 14.0, 15.0]), atol=1e-12, label="last column")


@check("im2col flattens each window in (channel, row, column) order")
def test_im2col_order(sub):
    got = has(sub, "im2col")
    x = np.arange(18.0).reshape(1, 2, 3, 3)
    col = got(x, 2, 1, 0)
    shape(col, (1, 8, 4), label="im2col output")
    # Window at output (0, 0): channel 0 holds 0, 1, 3, 4; channel 1 holds 9, 10, 12, 13.
    want = np.array([0.0, 1.0, 3.0, 4.0, 9.0, 10.0, 12.0, 13.0])
    close(col[0, :, 0], want, atol=1e-12, label="first window, all channels")


@check("im2col matches direct slicing with stride, padding, and a non-square image")
def test_im2col_slices(sub):
    got = has(sub, "im2col")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 2, 5, 6))
    k, stride, padding = 3, 2, 1
    col = got(x, k, stride, padding)
    h_out = (5 + 2 * padding - k) // stride + 1
    w_out = (6 + 2 * padding - k) // stride + 1
    shape(col, (2, 2 * k * k, h_out * w_out), label="im2col output")
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    for i in range(h_out):
        for j in range(w_out):
            patch = xp[:, :, i * stride : i * stride + k, j * stride : j * stride + k]
            close(
                col[:, :, i * w_out + j],
                patch.reshape(2, -1),
                atol=1e-12,
                label=f"column for output position ({i}, {j})",
            )


@check("conv2d_forward matches the loop reference at stride 1, no padding")
def test_forward_basic(sub):
    got = has(sub, "conv2d_forward")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 6, 6))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.3
    b = rng.normal(size=4) * 0.1
    y = got(x, w, b, 1, 0)
    shape(y, (2, 4, 4, 4), label="conv2d_forward output")
    close(y, reference_conv(x, w, b, 1, 0), atol=1e-8, label="stride 1, padding 0")


@check("conv2d_forward matches the reference with stride, padding, and a non-square image")
def test_forward_stride_padding(sub):
    got = has(sub, "conv2d_forward")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 2, 7, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    for stride, padding in [(2, 1), (1, 1), (3, 0)]:
        want = reference_conv(x, w, b, stride, padding)
        y = got(x, w, b, stride, padding)
        shape(y, want.shape, label=f"shape at stride={stride}, padding={padding}")
        close(y, want, atol=1e-8, label=f"stride={stride}, padding={padding}")


@check("conv2d_forward agrees with the loop version from the first chapter")
def test_forward_vs_naive(sub):
    forward = has(sub, "conv2d_forward")
    naive = has(sub, "conv2d_naive")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 3, 8, 8))
    w = rng.normal(size=(2, 3, 5, 5)) * 0.2
    b = rng.normal(size=2)
    close(forward(x, w, b, 1, 2), naive(x, w, b, 1, 2), atol=1e-8, label="5 by 5 kernels, padding 2")


@check("conv1x1_as_matmul is a convolution with a 1 by 1 kernel")
def test_conv1x1(sub):
    got = has(sub, "conv1x1_as_matmul")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 4, 5, 6))
    w = rng.normal(size=(3, 4, 1, 1))
    b = rng.normal(size=3)
    y = got(x, w, b)
    shape(y, (2, 3, 5, 6), label="conv1x1_as_matmul output")
    close(y, reference_conv(x, w, b, 1, 0), atol=1e-10, label="1 by 1 convolution")

    # At one pixel it is a dense layer on that pixel's channels.
    pixel = x[1, :, 2, 3]
    close(y[1, :, 2, 3], w[:, :, 0, 0] @ pixel + b, atol=1e-10, label="one pixel, as a dense layer")
