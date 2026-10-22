"""Flops against bytes: why generating one token waits on memory.

Fill in the three functions below. All quantities are plain numbers: flops and
bytes are counts, `bandwidth` is bytes per second, and `peak_flops` is flops
per second.
"""

import math

import numpy as np


def arithmetic_intensity(batch: int, bytes_per_elt: float) -> float:
    """Flops done per byte read when a weight matrix multiplies `batch` vectors.

    Every weight is read once. It then does one multiply and one add for each
    of the `batch` vectors. The width of the matrix does not appear.

    Args:
        batch: Number of vectors that share one read of the weights.
        bytes_per_elt: Bytes per weight, for example 2 for bf16.

    Returns:
        Flops per byte, as a float.
    """
    raise NotImplementedError("arithmetic_intensity")


def decode_seconds_per_token(
    n_params: float, bytes_per_elt: float, bandwidth: float
) -> float:
    """Seconds to read every weight once, which is the time of one decode step.

    This is the memory-bound estimate for batch 1. It ignores the cache and
    any arithmetic, so it is a lower bound on the true time.

    Args:
        n_params: Number of weights in the model.
        bytes_per_elt: Bytes per weight.
        bandwidth: Memory bandwidth in bytes per second.

    Returns:
        Seconds per token.
    """
    raise NotImplementedError("decode_seconds_per_token")


def min_batch_to_saturate(
    bytes_per_elt: float, peak_flops: float, bandwidth: float
) -> int:
    """Smallest whole batch whose arithmetic intensity reaches the ridge point.

    The ridge point is `peak_flops / bandwidth`, the flops per byte at which a
    chip stops waiting on memory and starts running out of arithmetic.

    Args:
        bytes_per_elt: Bytes per weight.
        peak_flops: Peak arithmetic rate in flops per second.
        bandwidth: Memory bandwidth in bytes per second.

    Returns:
        The smallest integer batch B with `arithmetic_intensity(B, bytes_per_elt)`
        at least the ridge point.
    """
    raise NotImplementedError("min_batch_to_saturate")


if __name__ == "__main__":
    n_params = 7e9          # a 7-billion-parameter model
    b = 2                   # bf16, two bytes per weight
    peak = 312e12           # flops per second
    bw = 2e12               # bytes per second

    print(f"intensity at batch 1: {arithmetic_intensity(1, b):.1f} flops per byte")
    print(f"ridge point: {peak / bw:.0f} flops per byte")
    t = decode_seconds_per_token(n_params, b, bw)
    print(f"one token reads the weights in {t * 1e3:.2f} ms ({1 / t:.0f} tokens per second)")
    print(f"batch needed to saturate the chip: {min_batch_to_saturate(b, peak, bw)}")
