"""Checks for the convolution and receptive field exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def reference_conv(x, w, b, stride, padding):
    """An independent reference built from numpy's sliding windows."""
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, w.shape[2:], axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    return np.einsum("nchwab,ocab->nohw", windows, w) + b[None, :, None, None]


@check("conv2d_single finds the edge in the 5 by 5 picture and does not flip the kernel")
def test_conv2d_single_finds_the_edge_in_the_5_by(sub):
    got = has(sub, "conv2d_single")
    x = np.array([[0, 0, 1, 1, 1]] * 5, dtype=float)
    edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    y = got(x, edge, 0.0)
    shape(y, (3, 3), label="conv2d_single output")
    close(y, np.array([[3.0, 3.0, 0.0]] * 3), atol=1e-12, label="edge response")
    got = has(sub, "conv2d_single")
    x = np.arange(1.0, 10.0).reshape(3, 3)
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = got(x, w, 0.5)
    # Top-left window [[1, 2], [4, 5]] gives 1*1 + 2*2 + 3*4 + 4*5 + 0.5 = 37.5.
    want = np.array([[37.5, 47.5], [67.5, 77.5]])
    close(y, want, atol=1e-12, label="conv2d_single on 1..9 with a (1, 2, 3, 4) kernel")


@check("conv2d_naive matches the reference with several channels and images, and handles stride and padding")
def test_conv2d_naive_matches_the_reference_with(sub):
    got = has(sub, "conv2d_naive")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 6, 6))
    w = rng.normal(size=(4, 3, 3, 3)) * 0.3
    b = rng.normal(size=4) * 0.1
    y = got(x, w, b, 1, 0)
    shape(y, (2, 4, 4, 4), label="conv2d_naive output")
    close(y, reference_conv(x, w, b, 1, 0), atol=1e-8, label="stride 1, padding 0")
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
def test_shifting_the_input_shifts_the_output_awa(sub):
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
def test_param_counts_reproduces_the_chapter_s_nu(sub):
    got = has(sub, "param_counts")
    photo = got(224 * 224 * 3, 1000, 3, 64, 3)
    equal(int(photo["dense"]), 150_529_000, label="dense layer on the photo")
    equal(int(photo["conv"]), 1_792, label="3 by 3 convolution with 64 filters on the photo")
    wide = got(1, 1, 64, 64, 3)
    equal(int(wide["conv"]), 36_928, label="3 by 3 convolution, 64 channels in and out")
    equal(int(wide["separable"]), 4_736, label="depthwise separable, 64 channels in and out")
    equal(int(got(6, 4, 1, 1, 3)["dense"]), 28, label="dense layer on a 6-pixel strip with 4 outputs")


@check("conv_output_shape matches the table, drops leftover cells, and keeps the size with padding (k - 1) / 2")
def test_conv_output_shape_matches_the_table_drop(sub):
    got = has(sub, "conv_output_shape")
    cases = [
        # (size, kernel_size, stride, padding, dilation, expected)
        (7, 3, 1, 0, 1, 5),
        (7, 3, 1, 1, 1, 7),
        (7, 3, 2, 1, 1, 4),
        (7, 3, 1, 0, 2, 3),
    ]
    for size, k, s, p, d, want in cases:
        value = got(size, k, s, p, d)
        if int(value) != want:
            fail(
                f"conv_output_shape(size={size}, kernel_size={k}, stride={s}, "
                f"padding={p}, dilation={d}) returned {value}, expected {want}"
            )
    got = has(sub, "conv_output_shape")
    # Windows start at 0 and 3 and cover cells 0 to 5; cell 6 is never read.
    equal(int(got(7, 3, 3, 0, 1)), 2, label="size 7, kernel 3, stride 3")
    equal(int(got(10, 3, 2, 0, 1)), 4, label="size 10, kernel 3, stride 2")
    equal(int(got(2, 3, 1, 0, 1)), 0, label="a kernel wider than the input fits zero times")
    got = has(sub, "conv_output_shape")
    for size in (5, 12, 32):
        for k in (1, 3, 5, 7):
            value = got(size, k, 1, (k - 1) // 2, 1)
            if int(value) != size:
                fail(f"size {size}, kernel {k}, padding {(k - 1) // 2}: got {value}, expected {size}")


@check("receptive_field_per_layer grows 3, 7, 15 for the chapter's stack and uses the jump of earlier layers")
def test_receptive_field_per_layer_grows_3_7_15_f(sub):
    got = has(sub, "receptive_field_per_layer")
    out = [int(v) for v in got([(3, 2), (3, 2), (3, 1)])]
    equal(out, [3, 7, 15], label="fields after each layer")
    got = has(sub, "receptive_field_per_layer")
    # Strides come last here, so they cannot widen anything yet.
    equal([int(v) for v in got([(3, 1), (3, 2), (3, 2)])], [3, 5, 9], label="stride-1 layer first")
    equal([int(v) for v in got([(3, 1), (3, 1)])], [3, 5], label="two 3 by 3 layers")
    equal([int(v) for v in got([(5, 1)])], [5], label="one 5 by 5 layer")


@check("max_pool_forward keeps the largest value of each window, per channel, and a one-pixel shift changes only some pooled values")
def test_max_pool_forward_keeps_the_largest_value(sub):
    got = has(sub, "max_pool_forward")
    grid = np.array(
        [[1, 3, 2, 5], [4, 9, 1, 0], [6, 2, 7, 1], [1, 1, 3, 2]], dtype=float
    )
    x = np.stack([grid, -grid])[None]  # shape (1, 2, 4, 4)
    out = got(x, 2, 2)
    shape(out, (1, 2, 2, 2), label="max_pool_forward output")
    close(out[0, 0], np.array([[9.0, 5.0], [6.0, 7.0]]), atol=1e-12, label="channel 0")
    close(out[0, 1], np.array([[-1.0, 0.0], [-1.0, -1.0]]), atol=1e-12, label="channel 1 (negated)")
    got = has(sub, "max_pool_forward")
    grid = np.array(
        [[1, 3, 2, 5], [4, 9, 1, 0], [6, 2, 7, 1], [1, 1, 3, 2]], dtype=float
    )
    shifted = np.zeros_like(grid)
    shifted[:, :-1] = grid[:, 1:]  # the picture moves one pixel left
    before = got(grid[None, None], 2, 2)[0, 0]
    after = got(shifted[None, None], 2, 2)[0, 0]
    close(after, np.array([[9.0, 5.0], [7.0, 2.0]]), atol=1e-12, label="pooled map after the shift")
    if np.array_equal(before, after):
        fail("the pooled map did not change at all; the bottom row should change")


@check("max_pool_forward handles overlapping windows and random data")
def test_max_pool_forward_handles_overlapping_win(sub):
    got = has(sub, "max_pool_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 9, 8))
    for pool, stride in [(3, 2), (2, 2), (3, 3)]:
        h_out = (9 - pool) // stride + 1
        w_out = (8 - pool) // stride + 1
        want = np.empty((2, 3, h_out, w_out))
        for i in range(h_out):
            for j in range(w_out):
                want[:, :, i, j] = x[
                    :, :, i * stride : i * stride + pool, j * stride : j * stride + pool
                ].reshape(2, 3, -1).max(axis=2)
        out = got(x, pool, stride)
        shape(out, want.shape, label=f"shape at pool={pool}, stride={stride}")
        close(out, want, atol=1e-12, label=f"pool={pool}, stride={stride}")
