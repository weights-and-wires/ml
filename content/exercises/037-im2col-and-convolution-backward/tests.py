"""Checks for the im2col and convolution backward exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


def reference_conv(x, w, b, stride, padding):
    """An independent reference built from numpy's sliding windows."""
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, w.shape[2:], axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    return np.einsum("nchwab,ocab->nohw", windows, w) + b[None, :, None, None]


@check("im2col has the right shape and flattens each window in (channel, row, column) order")
def test_im2col_layout(sub):
    got = has(sub, "im2col")
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    col = got(x, 2, 1, 0)
    shape(col, (1, 4, 9), label="im2col output")
    close(col[0, :, 0], np.array([0.0, 1.0, 4.0, 5.0]), atol=1e-12, label="first column")
    close(col[0, :, 8], np.array([10.0, 11.0, 14.0, 15.0]), atol=1e-12, label="last column")

    x = np.arange(18.0).reshape(1, 2, 3, 3)
    col = got(x, 2, 1, 0)
    shape(col, (1, 8, 4), label="im2col output with two channels")
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


@check("conv2d_forward matches the reference at several strides and paddings, on a non-square image")
def test_forward_reference(sub):
    got = has(sub, "conv2d_forward")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 6, 6))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.3
    b = rng.normal(size=4) * 0.1
    y = got(x, w, b, 1, 0)
    shape(y, (2, 4, 4, 4), label="conv2d_forward output")
    close(y, reference_conv(x, w, b, 1, 0), atol=1e-8, label="stride 1, padding 0")

    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 2, 7, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    for stride, padding in [(2, 1), (1, 1), (3, 0)]:
        want = reference_conv(x, w, b, stride, padding)
        y = got(x, w, b, stride, padding)
        shape(y, want.shape, label=f"shape at stride={stride}, padding={padding}")
        close(y, want, atol=1e-8, label=f"stride={stride}, padding={padding}")


@check("conv2d_forward agrees with the loop version")
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


@check("col2im adds overlapping windows: columns of ones count the covering windows")
def test_col2im_counts(sub):
    got = has(sub, "col2im")
    # A 3 by 3 image with 2 by 2 windows has 4 windows of 4 entries each.
    ones = np.ones((1, 4, 4))
    out = got(ones, (1, 1, 3, 3), 2, 1, 0)
    shape(out, (1, 1, 3, 3), label="col2im output")
    want = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    close(out[0, 0], want, atol=1e-12, label="windows covering each pixel")


@check("col2im is the transpose of im2col (adjoint identity)")
def test_col2im_adjoint(sub):
    col2im = has(sub, "col2im")
    im2col = has(sub, "im2col")
    rng = np.random.default_rng(0)
    for stride, padding, k in [(1, 0, 2), (1, 1, 3), (2, 1, 3), (2, 0, 3)]:
        x = rng.normal(size=(2, 3, 7, 6))
        cols = im2col(x, k, stride, padding)
        c = rng.normal(size=cols.shape)
        back = col2im(c, x.shape, k, stride, padding)
        shape(back, x.shape, label=f"col2im shape at stride={stride}, padding={padding}, k={k}")
        left = float(np.sum(cols * c))
        right = float(np.sum(x * back))
        if abs(left - right) > 1e-8 * max(1.0, abs(left)):
            fail(
                f"at stride={stride}, padding={padding}, k={k}: <im2col(x), c> = {left:.6f} "
                f"but <x, col2im(c)> = {right:.6f}; they must be equal"
            )


@check("window_coverage counts windows per pixel, including padding and skipped pixels")
def test_window_coverage(sub):
    got = has(sub, "window_coverage")
    out = got((1, 1, 3, 3), 2, 1, 0)
    close(out, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float), atol=1e-12, label="3x3, k=2")

    # Windows start at columns 0 and 3, so columns 2 and 5 are never read.
    out = got((1, 1, 2, 6), 2, 3, 0)
    close(
        out,
        np.array([[1, 1, 0, 1, 1, 0]] * 2, dtype=float),
        atol=1e-12,
        label="stride 3 skips pixels",
    )

    # With padding 1 and k=3, the middle of a 4 by 4 image is read by 9 windows.
    out = got((1, 1, 4, 4), 3, 1, 1)
    equal(int(out[1, 1]), 9, label="an interior pixel with k=3")
    equal(int(out[0, 0]), 4, label="a corner pixel with k=3 and padding 1")


@check("conv2d_backward's dx, dw, and db match finite differences")
def test_backward_finite_difference(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    dY = rng.normal(size=forward(x, w, b, 2, 1).shape)
    dx, dw, db = backward(dY, x, w, b, 2, 1)
    shape(dx, x.shape, label="dx")
    shape(dw, w.shape, label="dw")
    shape(db, b.shape, label="db")

    numeric_x = finite_difference(lambda a: np.sum(dY * forward(a, w, b, 2, 1)), x, eps=1e-5)
    close(dx, numeric_x, rtol=1e-3, atol=1e-4, label="dL/dx at stride 2, padding 1")
    numeric_w = finite_difference(lambda a: np.sum(dY * forward(x, a, b, 2, 1)), w, eps=1e-5)
    close(dw, numeric_w, rtol=1e-3, atol=1e-4, label="dL/dw at stride 2, padding 1")
    numeric_b = finite_difference(lambda a: np.sum(dY * forward(x, w, a, 2, 1)), b, eps=1e-5)
    close(db, numeric_b, rtol=1e-3, atol=1e-4, label="dL/db at stride 2, padding 1")


@check("conv2d_backward reproduces the hand example")
def test_backward_hand(sub):
    backward = has(sub, "conv2d_backward")
    x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    dx, dw, db = backward(np.ones((1, 1, 2, 2)), x, w, np.zeros(1), 1, 0)
    close(dw[0, 0], np.array([[12.0, 16.0], [24.0, 28.0]]), atol=1e-12, label="dw")
    close(db, np.array([4.0]), atol=1e-12, label="db")
    close(dx[0, 0], np.array([[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]]), atol=1e-12, label="dx")
