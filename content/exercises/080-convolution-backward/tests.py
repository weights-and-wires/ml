"""Checks for the convolution backward exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


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


@check("conv2d_backward's input gradient matches a finite difference")
def test_backward_dx(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    dY = rng.normal(size=forward(x, w, b, 2, 1).shape)

    def loss(x_arr):
        return np.sum(dY * forward(x_arr, w, b, 2, 1))

    numeric = finite_difference(loss, x, eps=1e-5)
    dx, _, _ = backward(dY, x, w, b, 2, 1)
    shape(dx, x.shape, label="dx")
    close(dx, numeric, rtol=1e-3, atol=1e-4, label="dL/dx at stride 2, padding 1")


@check("conv2d_backward's weight gradient matches a finite difference")
def test_backward_dw(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    dY = rng.normal(size=forward(x, w, b, 2, 1).shape)

    def loss(w_arr):
        return np.sum(dY * forward(x, w_arr, b, 2, 1))

    numeric = finite_difference(loss, w, eps=1e-5)
    _, dw, _ = backward(dY, x, w, b, 2, 1)
    shape(dw, w.shape, label="dw")
    close(dw, numeric, rtol=1e-3, atol=1e-4, label="dL/dw at stride 2, padding 1")


@check("conv2d_backward's bias gradient matches a finite difference")
def test_backward_db(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    dY = rng.normal(size=forward(x, w, b, 1, 0).shape)

    def loss(b_arr):
        return np.sum(dY * forward(x, w, b_arr, 1, 0))

    numeric = finite_difference(loss, b, eps=1e-5)
    _, _, db = backward(dY, x, w, b, 1, 0)
    shape(db, b.shape, label="db")
    close(db, numeric, rtol=1e-3, atol=1e-4, label="dL/db at stride 1, padding 0")


@check("conv2d_backward reproduces the chapter's hand example")
def test_backward_hand(sub):
    backward = has(sub, "conv2d_backward")
    x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    dx, dw, db = backward(np.ones((1, 1, 2, 2)), x, w, np.zeros(1), 1, 0)
    close(dw[0, 0], np.array([[12.0, 16.0], [24.0, 28.0]]), atol=1e-12, label="dw")
    close(db, np.array([4.0]), atol=1e-12, label="db")
    close(dx[0, 0], np.array([[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]]), atol=1e-12, label="dx")
