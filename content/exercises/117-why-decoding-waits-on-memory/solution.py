"""Reference solution: flops against bytes for one decode step."""

import math

import numpy as np


def arithmetic_intensity(batch: int, bytes_per_elt: float) -> float:
    # Per weight: 2 * batch flops (a multiply and an add per vector), one read.
    return 2.0 * batch / bytes_per_elt


def decode_seconds_per_token(
    n_params: float, bytes_per_elt: float, bandwidth: float
) -> float:
    return n_params * bytes_per_elt / bandwidth


def min_batch_to_saturate(
    bytes_per_elt: float, peak_flops: float, bandwidth: float
) -> int:
    ridge = peak_flops / bandwidth
    # Solve 2 * B / b >= ridge for B, then round up to a whole batch. The tiny
    # offset keeps a ridge that is an exact integer from rounding up by one.
    return int(math.ceil(ridge * bytes_per_elt / 2.0 - 1e-9))


if __name__ == "__main__":
    n_params = 7e9
    b = 2
    peak = 312e12
    bw = 2e12

    print(f"intensity at batch 1: {arithmetic_intensity(1, b):.1f} flops per byte")
    print(f"ridge point: {peak / bw:.0f} flops per byte")
    t = decode_seconds_per_token(n_params, b, bw)
    print(f"one token reads the weights in {t * 1e3:.2f} ms ({1 / t:.0f} tokens per second)")
    print(f"batch needed to saturate the chip: {min_batch_to_saturate(b, peak, bw)}")
