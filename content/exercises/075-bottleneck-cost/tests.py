"""Checks for the bottleneck-cost exercise."""

from mlcheck import almost, check, equal, fail, has


@check("conv_flops matches the 2x2 example from the chapter: 64 for 1x1 and 576 for 3x3")
def test_conv_small(sub):
    conv_flops = has(sub, "conv_flops")
    equal(conv_flops(4, 4, 1, 2, 2), 64, label="1x1 convolution, 4 -> 4 channels, 2x2 picture")
    equal(conv_flops(4, 4, 3, 2, 2), 576, label="3x3 convolution, 4 -> 4 channels, 2x2 picture")


@check("conv_flops is a product of height, width, both channel counts, and the kernel area")
def test_conv_general(sub):
    conv_flops = has(sub, "conv_flops")
    cases = [(3, 8, 3, 32, 32), (256, 256, 3, 14, 14), (64, 128, 1, 28, 14), (16, 16, 5, 7, 9)]
    for c_in, c_out, k, h, w in cases:
        want = h * w * c_in * c_out * k * k
        got = conv_flops(c_in, c_out, k, h, w)
        if int(got) != want:
            fail(f"conv_flops({c_in}, {c_out}, {k}, {h}, {w}) returned {got}, expected {want}")


@check("conv_flops and bottleneck_flops return plain Python ints")
def test_ints(sub):
    conv_flops = has(sub, "conv_flops")
    bottleneck_flops = has(sub, "bottleneck_flops")
    for name, value in (("conv_flops", conv_flops(8, 8, 3, 4, 4)), ("bottleneck_flops", bottleneck_flops(8, 2, 8, 4, 4))):
        if not isinstance(value, int) or isinstance(value, bool):
            fail(f"{name} should return an int, got {type(value).__name__}")


@check("bottleneck_flops matches the hand-computed count, including 256 -> 64 -> 256 at 14x14")
def test_bottleneck(sub):
    bottleneck_flops = has(sub, "bottleneck_flops")
    equal(bottleneck_flops(256, 64, 256, 14, 14), 13_647_872, label="256 -> 64 -> 256 at 14x14")
    cases = [(64, 16, 64, 8, 8), (128, 32, 256, 28, 28), (10, 3, 7, 5, 6)]
    for c_in, c_mid, c_out, h, w in cases:
        want = h * w * (c_in * c_mid + 9 * c_mid * c_mid + c_mid * c_out)
        got = bottleneck_flops(c_in, c_mid, c_out, h, w)
        if int(got) != want:
            fail(f"bottleneck_flops({c_in}, {c_mid}, {c_out}, {h}, {w}) returned {got}, expected {want}")


@check("The factor of 9 matters: the 3x3 step dominates a bottleneck with a wide middle")
def test_nine(sub):
    bottleneck_flops = has(sub, "bottleneck_flops")
    got = bottleneck_flops(16, 128, 16, 1, 1)
    want = 16 * 128 + 9 * 128 * 128 + 128 * 16
    if got != want:
        fail(f"bottleneck_flops(16, 128, 16, 1, 1) returned {got}, expected {want}; the middle convolution is 3x3, so it carries a factor of 9")


@check("bottleneck_savings is about 17 for 256 channels at 14x14")
def test_savings_value(sub):
    savings = has(sub, "bottleneck_savings")
    value = savings(256, 14, 14)
    want = 231_211_008 / 13_647_872
    almost(value, want, tol=1e-9, label="savings at 256 channels, 14x14")
    if not (16.0 < value < 18.0):
        fail(f"the bottleneck should be about 17 times cheaper, got {value:.3f}")


@check("The savings ratio is 288/17 at every picture size and every width divisible by 4")
def test_savings_constant(sub):
    savings = has(sub, "bottleneck_savings")
    for c, h, w in [(64, 8, 8), (128, 28, 28), (256, 7, 7), (512, 14, 3)]:
        almost(savings(c, h, w), 288 / 17, tol=1e-9, label=f"savings at c={c}, {h}x{w}")
