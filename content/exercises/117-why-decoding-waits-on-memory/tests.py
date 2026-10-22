"""Checks for the flops-against-bytes exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


@check("arithmetic_intensity is 1 flop per byte for batch 1 in bf16")
def test_intensity_hand(sub):
    got = has(sub, "arithmetic_intensity")
    almost(got(1, 2), 1.0, tol=1e-12, label="arithmetic_intensity(1, 2)")
    almost(got(156, 2), 156.0, tol=1e-9, label="arithmetic_intensity(156, 2)")
    almost(got(8, 4), 4.0, tol=1e-12, label="arithmetic_intensity(8, 4)")


@check("arithmetic_intensity doubles when the batch doubles and halves when bytes double")
def test_intensity_scaling(sub):
    got = has(sub, "arithmetic_intensity")
    for batch in (1, 3, 17, 64):
        almost(got(2 * batch, 2), 2 * got(batch, 2), tol=1e-9, label=f"batch {batch}")
        almost(got(batch, 4), got(batch, 2) / 2, tol=1e-9, label=f"bytes at batch {batch}")


@check("decode_seconds_per_token reads 14 GB at 2 TB/s in 7 ms")
def test_decode_time(sub):
    got = has(sub, "decode_seconds_per_token")
    almost(got(7e9, 2, 2e12), 0.007, tol=1e-9, label="7B parameters in bf16")
    almost(got(7e9, 1, 2e12), 0.0035, tol=1e-9, label="7B parameters in 8-bit")
    almost(got(14e9, 2, 2e12), 0.014, tol=1e-9, label="14B parameters in bf16")


@check("decode_seconds_per_token does not depend on the arithmetic rate")
def test_decode_time_bandwidth(sub):
    got = has(sub, "decode_seconds_per_token")
    slow = got(7e9, 2, 1e12)
    fast = got(7e9, 2, 4e12)
    almost(slow / fast, 4.0, tol=1e-9, label="4x the bandwidth gives 1/4 of the time")


@check("min_batch_to_saturate gives 156 for the 312 TFLOPs, 2 TB/s chip in bf16")
def test_min_batch_hand(sub):
    got = has(sub, "min_batch_to_saturate")
    equal(got(2, 312e12, 2e12), 156, label="bf16")
    equal(got(4, 312e12, 2e12), 312, label="fp32")
    equal(got(1, 312e12, 2e12), 78, label="8-bit")


@check("min_batch_to_saturate rounds a fractional ridge point up")
def test_min_batch_rounding(sub):
    got = has(sub, "min_batch_to_saturate")
    # ridge = 100e12 / 3e12 = 33.33 flops per byte, and b = 2 needs B >= 33.33.
    equal(got(2, 100e12, 3e12), 34, label="ridge 33.33")
    if not isinstance(got(2, 100e12, 3e12), (int, np.integer)):
        fail("min_batch_to_saturate must return an int")


@check("The returned batch reaches the ridge and one fewer does not")
def test_min_batch_boundary(sub):
    intensity = has(sub, "arithmetic_intensity")
    got = has(sub, "min_batch_to_saturate")
    rng = np.random.default_rng(0)
    for _ in range(25):
        peak = rng.uniform(50, 900) * 1e12
        bw = rng.uniform(0.5, 4.0) * 1e12
        b = float(rng.choice([1, 2, 4]))
        ridge = peak / bw
        batch = got(b, peak, bw)
        if intensity(batch, b) < ridge:
            fail(f"batch {batch} has intensity {intensity(batch, b):.3f}, below the ridge {ridge:.3f}")
        if batch > 1 and intensity(batch - 1, b) >= ridge:
            fail(f"batch {batch - 1} already reaches the ridge {ridge:.3f}, so {batch} is not the smallest")
