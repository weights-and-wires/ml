"""Checks for the convolution exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def direct_conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, padding: int) -> np.ndarray:
    """A quadruple-loop reference convolution, independent of im2col."""
    n, c_in, h, w_dim = x.shape
    c_out, _, k, _ = w.shape
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w_dim + 2 * padding - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    out = np.zeros((n, c_out, h_out, w_out))
    for nn in range(n):
        for co in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    acc = b[co]
                    for ci in range(c_in):
                        for dh in range(k):
                            for dw in range(k):
                                acc += w[co, ci, dh, dw] * xp[nn, ci, i * stride + dh, j * stride + dw]
                    out[nn, co, i, j] = acc
    return out


@check("conv_output_shape matches the formula on known stride and padding cases")
def test_conv_output_shape(sub):
    got = has(sub, "conv_output_shape")
    cases = [
        # (size, kernel_size, stride, padding, dilation, expected)
        (7, 3, 1, 0, 1, 5),
        (7, 3, 1, 1, 1, 7),
        (7, 3, 2, 1, 1, 4),
        (10, 3, 2, 0, 1, 4),
        (9, 3, 3, 0, 1, 3),
        (7, 3, 1, 0, 2, 3),
    ]
    for size, k, s, p, d, want in cases:
        value = got(size, k, s, p, d)
        if int(value) != want:
            fail(
                f"conv_output_shape(size={size}, kernel_size={k}, stride={s}, "
                f"padding={p}, dilation={d}) returned {value}, expected {want}"
            )


@check("im2col gathers the same patches a direct slice would")
def test_im2col(sub):
    got = has(sub, "im2col")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 2, 4, 4))
    k, stride, padding = 2, 1, 0
    col = got(x, k, stride, padding)
    shape(col, (1, 2 * k * k, 3 * 3), label="im2col output")

    w_out = 3
    for i in range(3):
        for j in range(3):
            patch = x[0, :, i : i + k, j : j + k].reshape(-1)
            column = col[0, :, i * w_out + j]
            close(column, patch, atol=1e-10, label=f"patch at output position ({i},{j})")


@check("conv2d_forward matches an independent quadruple-loop reference")
def test_conv2d_forward(sub):
    got = has(sub, "conv2d_forward")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 2, 6, 6))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=(3,)) * 0.1

    for stride, padding in [(1, 0), (2, 1)]:
        y = got(x, w, b, stride, padding)
        want = direct_conv2d(x, w, b, stride, padding)
        close(y, want, atol=1e-8, label=f"stride={stride}, padding={padding}")


@check("conv2d_backward's input gradient matches a finite difference")
def test_conv2d_backward_dx(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=(3,)) * 0.1
    stride, padding = 2, 1
    dY = rng.normal(size=forward(x, w, b, stride, padding).shape)

    def loss(x_flat):
        return np.sum(dY * forward(x_flat, w, b, stride, padding))

    numeric = finite_difference(loss, x, eps=1e-5)
    dx, _, _ = backward(dY, x, w, b, stride, padding)
    shape(dx, x.shape, label="dx")
    close(dx, numeric, rtol=1e-3, atol=1e-4, label="dL/dx")


@check("conv2d_backward's weight gradient matches a finite difference")
def test_conv2d_backward_dw(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=(3,)) * 0.1
    stride, padding = 2, 1
    dY = rng.normal(size=forward(x, w, b, stride, padding).shape)

    def loss(w_flat):
        return np.sum(dY * forward(x, w_flat, b, stride, padding))

    numeric = finite_difference(loss, w, eps=1e-5)
    _, dw, _ = backward(dY, x, w, b, stride, padding)
    shape(dw, w.shape, label="dw")
    close(dw, numeric, rtol=1e-3, atol=1e-4, label="dL/dw")


@check("conv2d_backward's bias gradient matches a finite difference")
def test_conv2d_backward_db(sub):
    forward = has(sub, "conv2d_forward")
    backward = has(sub, "conv2d_backward")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=(3,)) * 0.1
    stride, padding = 2, 1
    dY = rng.normal(size=forward(x, w, b, stride, padding).shape)

    def loss(b_flat):
        return np.sum(dY * forward(x, w, b_flat, stride, padding))

    numeric = finite_difference(loss, b, eps=1e-5)
    _, _, db = backward(dY, x, w, b, stride, padding)
    shape(db, b.shape, label="db")
    close(db, numeric, rtol=1e-3, atol=1e-4, label="dL/db")


@check("Shifting the input shifts the output by the same amount")
def test_translation_equivariance(sub):
    forward = has(sub, "conv2d_forward")
    rng = np.random.default_rng(6)
    x = rng.normal(size=(1, 2, 8, 8))
    w = rng.normal(size=(2, 2, 3, 3)) * 0.4
    b = rng.normal(size=(2,)) * 0.1

    y = forward(x, w, b, stride=1, padding=0)

    x_shifted = np.zeros_like(x)
    x_shifted[:, :, 1:, 1:] = x[:, :, :-1, :-1]
    y_shifted = forward(x_shifted, w, b, stride=1, padding=0)

    close(
        y_shifted[:, :, 1:, 1:],
        y[:, :, :-1, :-1],
        atol=1e-8,
        label="interior of the shifted output against the un-shifted output",
    )


@check("max_pool_forward matches a direct windowed maximum")
def test_max_pool_forward(sub):
    got = has(sub, "max_pool_forward")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 3, 8, 8))
    pool_size, stride = 2, 2
    out = got(x, pool_size, stride)
    shape(out, (2, 3, 4, 4), label="max_pool_forward output")

    want = np.zeros_like(out)
    for i in range(4):
        for j in range(4):
            window = x[:, :, i * stride : i * stride + pool_size, j * stride : j * stride + pool_size]
            want[:, :, i, j] = window.max(axis=(2, 3))
    close(out, want, atol=1e-10, label="pooled values")


@check("receptive_field matches the hand-computed value for a three-layer stack")
def test_receptive_field(sub):
    got = has(sub, "receptive_field")
    # k=3,s=2 then k=3,s=2 then k=3,s=1: r goes 1 -> 3 -> 7 -> 15.
    value = got([(3, 2), (3, 2), (3, 1)])
    if int(value) != 15:
        fail(f"receptive_field([(3,2),(3,2),(3,1)]) returned {value}, expected 15")

    # A single layer's receptive field is just its kernel size.
    value_single = got([(5, 1)])
    if int(value_single) != 5:
        fail(f"receptive_field([(5,1)]) returned {value_single}, expected 5")
