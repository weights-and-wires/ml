"""Checks for the cost of training exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


@check("training_flops is 6ND on the worked example")
def test_flops_worked(sub):
    got = has(sub, "training_flops")
    equal(int(got(1_000_000, 10_000)), 6 * 10**10, label="a million weights on ten thousand tokens")
    equal(int(got(357, 42)), 6 * 357 * 42, label="small numbers")


@check("training_flops at the scale of a billion weights, and linear in both arguments")
def test_flops_scale(sub):
    got = has(sub, "training_flops")
    almost(float(got(10**9, 2 * 10**10)), 1.2e20, tol=1e8, label="1e9 weights on 2e10 tokens")
    almost(float(got(10**8, 10**10)), 6e18, tol=1e6, label="1e8 weights on 1e10 tokens")
    almost(float(got(2000, 300)) / float(got(1000, 300)), 2.0, tol=1e-12, label="doubling the weights doubles the cost")
    almost(float(got(1000, 600)) / float(got(1000, 300)), 2.0, tol=1e-12, label="doubling the tokens doubles the cost")


@check("training_memory_bytes defaults to four fp32 copies, 16 bytes per weight")
def test_memory_defaults(sub):
    got = has(sub, "training_memory_bytes")
    equal(int(got(1_000_000)), 16_000_000, label="a million weights")
    equal(int(got(10**9)), 16 * 10**9, label="a billion weights")


@check("training_memory_bytes responds to the number size, the copy count, and the weight count")
def test_memory_variants(sub):
    got = has(sub, "training_memory_bytes")
    equal(int(got(1_000_000, bytes_per_number=2)), 8_000_000, label="2-byte numbers")
    equal(int(got(1_000_000, copies=2)), 8_000_000, label="plain descent, two copies")
    equal(int(got(1_000_000, 2, 3)), 6_000_000, label="positional arguments")
    equal(int(got(3_000)), 3 * int(got(1_000)), label="linear in the weight count")


@check("Without scaling, a gradient of 1e-8 rounds to zero in fp16")
def test_vanishes(sub):
    got = has(sub, "scaled_grad_survives")
    almost(float(got(1e-8, 1.0)), 0.0, tol=0.0, label="1e-8 at scale 1")


@check("A scale of 1024 brings 1e-8 back to within 1 percent")
def test_scaled(sub):
    got = has(sub, "scaled_grad_survives")
    out = float(got(1e-8, 1024.0))
    if abs(out - 1e-8) > 1e-10:
        fail(f"1e-8 at scale 1024 came back as {out:.4e}, expected within 1 percent of 1e-8")
    arr = got(np.array([1e-8, 3e-8, 2.0]), 1024.0)
    equal(tuple(np.shape(arr)), (3,), label="the result keeps the shape of an array input")
    close(arr, np.array([1e-8, 3e-8, 2.0]), rtol=0.01, atol=0, label="an array of gradients")


@check("An ordinary gradient is unharmed by scaling")
def test_ordinary(sub):
    got = has(sub, "scaled_grad_survives")
    close(float(got(3e-4, 1024.0)), 3e-4, rtol=0.005, atol=0, label="3e-4 at scale 1024")
    close(float(got(0.37, 8.0)), 0.37, rtol=0.001, atol=0, label="0.37 at scale 8")


@check("A scale that is too large overflows to infinity")
def test_overflow(sub):
    got = has(sub, "scaled_grad_survives")
    out = float(got(1.0, 1e5))
    if not np.isinf(out):
        fail(f"1.0 at scale 1e5 is 1e5 in fp16, above the largest number 65504, so it should be infinity; got {out}")
    out2 = float(got(1e-3, 65536.0))
    if np.isinf(out2) or abs(out2 - 1e-3) > 1e-5:
        fail(f"1e-3 at scale 65536 is about 65.5, which fits in fp16, but came back as {out2}")
