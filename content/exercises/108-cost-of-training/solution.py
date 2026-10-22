"""Reference solution for the cost of training exercise."""

import numpy as np


def training_flops(params: int, tokens: int) -> float:
    """The 6ND estimate of the flops in a training run.

    One token costs about 2 flops per weight forward and 4 backward.

    Args:
        params: N, the number of weights. This estimate counts every weight.
        tokens: D, the number of training tokens.

    Returns:
        About 6 * params * tokens.
    """
    return 6 * params * tokens


def training_memory_bytes(params: int, bytes_per_number: int = 4, copies: int = 4) -> int:
    """Bytes needed for `copies` numbers per weight.

    The defaults describe AdamW in fp32: weights, gradients, the average of
    the gradient (m), and the average of its square (v), 4 bytes each. The
    estimate leaves out activations.

    Args:
        params: N, the number of weights.
        bytes_per_number: The size of one number: 4 for fp32, 2 for fp16.
        copies: How many numbers are kept per weight.

    Returns:
        The total bytes, params * bytes_per_number * copies.
    """
    return params * bytes_per_number * copies


def fp16_roundtrip(x):
    """Given: cast to float16 and back to float64, to show what survives."""
    with np.errstate(over="ignore", under="ignore"):
        return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def scaled_grad_survives(true_grad, scale: float):
    """Scale a gradient up, store it in fp16, then scale it back down.

    Multiplying the loss by `scale` multiplies every gradient by `scale`. The
    scaled gradient is stored in float16 (use `fp16_roundtrip`). Dividing by
    `scale` afterwards, in float64 here, recovers the gradient.

    Args:
        true_grad: A gradient value or an array of them.
        scale: The loss scale.

    Returns:
        The recovered gradient, with the same shape as `true_grad`. It is 0
        where fp16 could not store the scaled value, and infinity where the
        scaled value was too large for fp16.
    """
    stored = fp16_roundtrip(np.asarray(true_grad, dtype=np.float64) * scale)
    return stored / scale


if __name__ == "__main__":
    n, d = 100_000_000, 10_000_000_000
    flops = training_flops(n, d)
    print(f"flops for {n:.0e} weights on {d:.0e} tokens: {flops:.2e}")
    print(f"hours at 1e14 flops per second: {flops / 1e14 / 3600:.1f}")
    print(f"memory in fp32: {training_memory_bytes(n) / 1e9:.1f} GB")
    big = training_flops(1_000_000_000, 20_000_000_000)
    print(f"1e9 weights on 2e10 tokens: {big:.2e} flops, {big / 1e14 / 86400:.1f} days")

    print("\nscale    stored in fp16 and recovered")
    for scale in (1, 1024, 65536):
        print(f"{scale:6d}   {float(scaled_grad_survives(1e-8, scale)):.4e}")
    print("a gradient of 1.0 at scale 1e5:", float(scaled_grad_survives(1.0, 1e5)))
