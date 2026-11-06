"""Checks for the sliding-a-filter exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def reference_conv(x, w, b, stride, padding):
    """An independent reference built from numpy's sliding windows."""
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, w.shape[2:], axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    return np.einsum("nchwab,ocab->nohw", windows, w) + b[None, :, None, None]


@check("conv2d_single finds the edge in the chapter's 5 by 5 picture")
def test_single_edge(sub):
    got = has(sub, "conv2d_single")
    x = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)
    edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    y = got(x, edge, 0.0)
    shape(y, (3, 3), label="conv2d_single output")
    close(y, np.array([[3.0, 3.0, 0.0]] * 3), atol=1e-12, label="edge response")


@check("conv2d_single does not flip the kernel")
def test_single_no_flip(sub):
    got = has(sub, "conv2d_single")
    x = np.arange(1.0, 10.0).reshape(3, 3)
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = got(x, w, 0.5)
    # Top-left window [[1, 2], [4, 5]] gives 1*1 + 2*2 + 3*4 + 4*5 + 0.5 = 37.5.
    want = np.array([[37.5, 47.5], [67.5, 77.5]])
    close(y, want, atol=1e-12, label="conv2d_single on 1..9 with a (1, 2, 3, 4) kernel")


@check("conv2d_single matches a sliding-window reference on random data")
def test_single_random(sub):
    got = has(sub, "conv2d_single")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 9))
    w = rng.normal(size=(3, 3))
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3))
    want = np.tensordot(windows, w, axes=([2, 3], [0, 1])) - 0.7
    close(got(x, w, -0.7), want, atol=1e-10, label="conv2d_single")


@check("conv2d_naive matches the reference with several channels and images")
def test_naive_basic(sub):
    got = has(sub, "conv2d_naive")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 6, 6))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.3
    b = rng.normal(size=4) * 0.1
    y = got(x, w, b, 1, 0)
    shape(y, (2, 4, 4, 4), label="conv2d_naive output")
    close(y, reference_conv(x, w, b, 1, 0), atol=1e-8, label="stride 1, padding 0")


@check("conv2d_naive handles stride and padding")
def test_naive_stride_padding(sub):
    got = has(sub, "conv2d_naive")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 2, 7, 6))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    for stride, padding in [(2, 1), (1, 1), (3, 0)]:
        y = got(x, w, b, stride, padding)
        want = reference_conv(x, w, b, stride, padding)
        shape(y, want.shape, label=f"output shape at stride={stride}, padding={padding}")
        close(y, want, atol=1e-8, label=f"stride={stride}, padding={padding}")


@check("Shifting the input shifts the output (away from the border)")
def test_equivariance(sub):
    got = has(sub, "conv2d_naive")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 2, 8, 8))
    w = rng.normal(size=(2, 2, 3, 3)) * 0.4
    b = rng.normal(size=2) * 0.1
    y = got(x, w, b, 1, 0)

    shifted = np.zeros_like(x)
    shifted[:, :, 1:, 1:] = x[:, :, :-1, :-1]
    y_shifted = got(shifted, w, b, 1, 0)
    close(
        y_shifted[:, :, 1:, 1:],
        y[:, :, :-1, :-1],
        atol=1e-8,
        label="shifted output against the original output, interior only",
    )


@check("param_counts reproduces the chapter's numbers")
def test_param_counts(sub):
    got = has(sub, "param_counts")
    photo = got(224 * 224 * 3, 1000, 3, 64, 3)
    equal(int(photo["dense"]), 150_529_000, label="dense layer on the photo")
    equal(int(photo["conv"]), 1_792, label="3 by 3 convolution with 64 filters on the photo")
    wide = got(1, 1, 64, 64, 3)
    equal(int(wide["conv"]), 36_928, label="3 by 3 convolution, 64 channels in and out")
    equal(int(wide["separable"]), 4_736, label="depthwise separable, 64 channels in and out")
    equal(int(got(6, 4, 1, 1, 3)["dense"]), 28, label="dense layer on a 6-pixel strip with 4 outputs")
