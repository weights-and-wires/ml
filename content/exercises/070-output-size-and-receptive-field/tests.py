"""Checks for the output-size and receptive-field exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


@check("conv_output_shape matches the chapter's four table rows")
def test_table(sub):
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


@check("conv_output_shape drops leftover cells when the window does not fit evenly")
def test_floor(sub):
    got = has(sub, "conv_output_shape")
    # Windows start at 0 and 3 and cover cells 0 to 5; cell 6 is never read.
    equal(int(got(7, 3, 3, 0, 1)), 2, label="size 7, kernel 3, stride 3")
    equal(int(got(10, 3, 2, 0, 1)), 4, label="size 10, kernel 3, stride 2")
    equal(int(got(2, 3, 1, 0, 1)), 0, label="a kernel wider than the input fits zero times")


@check("Padding (k - 1) / 2 keeps the size for odd kernels at stride 1")
def test_same_padding(sub):
    got = has(sub, "conv_output_shape")
    for size in (5, 12, 32):
        for k in (1, 3, 5, 7):
            value = got(size, k, 1, (k - 1) // 2, 1)
            if int(value) != size:
                fail(f"size {size}, kernel {k}, padding {(k - 1) // 2}: got {value}, expected {size}")


@check("receptive_field_per_layer grows 3, 7, 15 for the chapter's stack")
def test_fields_stack(sub):
    got = has(sub, "receptive_field_per_layer")
    out = [int(v) for v in got([(3, 2), (3, 2), (3, 1)])]
    equal(out, [3, 7, 15], label="fields after each layer")


@check("receptive_field_per_layer uses the jump of the earlier layers, not this one")
def test_fields_order(sub):
    got = has(sub, "receptive_field_per_layer")
    # Strides come last here, so they cannot widen anything yet.
    equal([int(v) for v in got([(3, 1), (3, 2), (3, 2)])], [3, 5, 9], label="stride-1 layer first")
    equal([int(v) for v in got([(3, 1), (3, 1)])], [3, 5], label="two 3 by 3 layers")
    equal([int(v) for v in got([(5, 1)])], [5], label="one 5 by 5 layer")


@check("max_pool_forward keeps the largest value of each window, per channel")
def test_pool_example(sub):
    got = has(sub, "max_pool_forward")
    grid = np.array(
        [[1, 3, 2, 5], [4, 9, 1, 0], [6, 2, 7, 1], [1, 1, 3, 2]], dtype=float
    )
    x = np.stack([grid, -grid])[None]  # shape (1, 2, 4, 4)
    out = got(x, 2, 2)
    shape(out, (1, 2, 2, 2), label="max_pool_forward output")
    close(out[0, 0], np.array([[9.0, 5.0], [6.0, 7.0]]), atol=1e-12, label="channel 0")
    close(out[0, 1], np.array([[-1.0, 0.0], [-1.0, -1.0]]), atol=1e-12, label="channel 1 (negated)")


@check("max_pool_forward handles overlapping windows and random data")
def test_pool_overlap(sub):
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


@check("A one-pixel shift leaves some pooled values unchanged and changes others")
def test_pool_shift(sub):
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
