"""Merge an adapter into the weight.

Fill in the three functions below. `lora_forward` is already written: it is the
function you wrote in the chapter "A low-rank update: build the LoRA layer".
"""

import numpy as np


# Written in the chapter "A low-rank update: build the LoRA layer".
def lora_forward(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    base = x @ W.T
    update = (x @ A.T) @ B.T
    return base + (alpha / r) * update


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
