"""LoRA: build the layer, then merge it into the weight.

Fill in the six functions below. W is the frozen weight, shape (d_out, d_in),
so the layer computes x @ W.T. A is (r, d_in) and B is (d_out, r).
"""

import numpy as np


def lora_parameter_count(d_in: int, d_out: int, r: int) -> int:
    """Return the number of trained numbers in A and B together.

    Args:
        d_in: Input width.
        d_out: Output width.
        r: The rank.

    Returns:
        r * (d_in + d_out), as a Python int.
    """
    raise NotImplementedError("lora_parameter_count")


def lora_init(d_in: int, d_out: int, r: int, seed: int) -> tuple:
    """Return a starting pair (A, B) for a LoRA layer.

    A is drawn from a normal distribution with standard deviation
    1 / sqrt(d_in), using np.random.default_rng(seed). B is all zeros.

    Args:
        d_in: Input width.
        d_out: Output width.
        r: The rank.
        seed: Seed for the random generator.

    Returns:
        A with shape (r, d_in) and B with shape (d_out, r).
    """
    raise NotImplementedError("lora_init")


def lora_forward(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    """Return the adapted layer's output x W^T + (alpha / r) (x A^T) B^T.

    Args:
        x: Inputs, shape (n, d_in).
        W: Frozen weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The scale numerator.
        r: The rank.

    Returns:
        The outputs, shape (n, d_out).
    """
    raise NotImplementedError("lora_forward")


def merge_lora(
    W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    """Return the merged weight W + (alpha / r) * (B @ A).

    Args:
        W: Frozen weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The scale numerator.
        r: The rank.

    Returns:
        The merged weight, shape (d_out, d_in).
    """
    raise NotImplementedError("merge_lora")


def extra_multiplies(d_in: int, d_out: int, r: int) -> int:
    """Return the multiplies the unmerged adapter adds per example.

    Args:
        d_in: Input width.
        d_out: Output width.
        r: The rank.

    Returns:
        The count of multiplies in (x @ A.T) @ B.T for one example, as an int.
    """
    raise NotImplementedError("extra_multiplies")


def merged_matches_split(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> float:
    """Return the largest absolute gap between merged and split outputs.

    Args:
        x: Inputs, shape (n, d_in).
        W: Frozen weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The scale numerator.
        r: The rank.

    Returns:
        max |x @ merge_lora(...).T - lora_forward(...)|, as a float.
    """
    raise NotImplementedError("merged_matches_split")


if __name__ == "__main__":
    # The 2 by 2 example from the chapter.
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    print("hand example (expect [7, 1]):", lora_forward(x, W, A, B, alpha=2.0, r=1))

    # A freshly initialised layer matches the frozen layer.
    rng = np.random.default_rng(0)
    d_in, d_out, r = 64, 48, 4
    W = rng.normal(size=(d_out, d_in))
    A, B = lora_init(d_in, d_out, r, seed=1)
    x = rng.normal(size=(5, d_in))
    gap = np.abs(lora_forward(x, W, A, B, 8.0, r) - x @ W.T).max()
    print("step-zero difference from the frozen layer:", gap)

    full = d_in * d_out
    small = lora_parameter_count(d_in, d_out, r)
    print(f"trained numbers: full {full:,}, LoRA {small:,} ({full / small:.1f} times fewer)")
    print("4096 x 4096 at rank 8:", lora_parameter_count(4096, 4096, 8))

    # Merging.
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    print("merged weight (expect [[3, 4], [0, 1]]):")
    print(merge_lora(W, A, B, 2.0, 1))

    rng = np.random.default_rng(0)
    d_in, d_out, r = 64, 48, 4
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in)) / np.sqrt(d_in)
    B = rng.normal(size=(d_out, r))
    x = rng.normal(size=(5, d_in))
    print("largest gap, merged vs split:", merged_matches_split(x, W, A, B, 8.0, r))

    extra = extra_multiplies(4096, 4096, 8)
    print(f"extra multiplies at 4096 x 4096, r = 8: {extra:,} ({extra / 4096**2:.2%} of the base)")
