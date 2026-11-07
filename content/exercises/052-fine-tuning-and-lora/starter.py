"""Fine-tuning on a budget: the memory bill, a LoRA layer, and what low rank reaches.

Fill in the nine functions below. W is the frozen weight, shape (d_out, d_in),
so a layer computes x @ W.T. A is (r, d_in) and B is (d_out, r).
"""

import numpy as np


def adam_bytes_breakdown() -> dict:
    """Return the bytes each trained parameter carries, piece by piece.

    Returns:
        A dict with exactly these keys, each mapped to an int byte count:
        "weight" (the 16-bit working copy), "gradient" (16-bit),
        "master_weight" (the 32-bit copy), "first_average" (Adam's m, 32-bit),
        and "second_average" (Adam's v, 32-bit).
    """
    raise NotImplementedError("adam_bytes_breakdown")


def trainable_fraction_memory(total_params: int, trained_params: int) -> int:
    """Return the bytes when only some parameters are trained.

    A trained parameter costs 16 bytes. A frozen parameter costs 2 bytes,
    because its weight must still sit in memory.

    Args:
        total_params: All parameters in the model.
        trained_params: How many of them are trained.

    Returns:
        The bytes needed, as an int.

    Raises:
        ValueError: If trained_params is larger than total_params.
    """
    raise NotImplementedError("trainable_fraction_memory")


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


def best_rank_approximation(delta_w: np.ndarray, r: int) -> np.ndarray:
    """Return the closest rank-r matrix to delta_w in Frobenius norm.

    Args:
        delta_w: A matrix, shape (d_out, d_in).
        r: The rank to keep, between 1 and min(delta_w.shape).

    Returns:
        The matrix keeping the r largest singular values, same shape as delta_w.
    """
    raise NotImplementedError("best_rank_approximation")


def energy_captured(delta_w: np.ndarray, r: int) -> float:
    """Return the share of squared singular values kept at rank r.

    Args:
        delta_w: A nonzero matrix, shape (d_out, d_in).
        r: The rank to keep, between 0 and min(delta_w.shape).

    Returns:
        (s_1^2 + ... + s_r^2) / (sum of all s_i^2), a float between 0 and 1.
    """
    raise NotImplementedError("energy_captured")


def lora_gradients(
    W: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
    r: int,
    target: np.ndarray,
) -> tuple:
    """Return the gradients of 0.5 * ||W_eff - target||^2 with respect to A and B.

    W_eff is merge_lora(W, A, B, alpha, r), and ||.|| is the Frobenius norm.

    Args:
        W: Frozen weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The scale numerator.
        r: The rank.
        target: The weight to match, shape (d_out, d_in).

    Returns:
        (dA, dB) with the shapes of A and B.
    """
    raise NotImplementedError("lora_gradients")


if __name__ == "__main__":
    pieces = adam_bytes_breakdown()
    print("bytes per trained parameter:", pieces, "->", sum(pieces.values()))
    n = 7_000_000_000
    frozen = trainable_fraction_memory(n, n // 100)
    print(f"train 1% of {n:,} and freeze the rest: {frozen / 1e9:.2f} GB")
    print("4096 x 4096 at rank 8 trains:", lora_parameter_count(4096, 4096, 8), "numbers")

    # The 2 by 2 example from the chapter.
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    print("hand example (expect [7, 1]):", lora_forward(x, W, A, B, alpha=2.0, r=1))
    print("merged weight (expect [[3, 4], [0, 1]]):")
    print(merge_lora(W, A, B, 2.0, 1))

    # A freshly initialised layer matches the frozen layer.
    rng = np.random.default_rng(0)
    d_in, d_out, r = 64, 48, 4
    W2 = rng.normal(size=(d_out, d_in))
    A2, B2 = lora_init(d_in, d_out, r, seed=1)
    x2 = rng.normal(size=(5, d_in))
    gap = np.abs(lora_forward(x2, W2, A2, B2, 8.0, r) - x2 @ W2.T).max()
    print("step-zero difference from the frozen layer:", gap)

    # Truncation and training.
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    approx = best_rank_approximation(M, 1)
    print("best rank-1 approximation of [[2, 1], [1, 2]]:")
    print(approx)
    print("error:", np.linalg.norm(M - approx), " energy kept:", energy_captured(M, 1))

    target = np.array([[1.0, 1.0], [2.0, 0.0]])
    dA, dB = lora_gradients(W, A, B, 2.0, 1, target)
    print("dA (expect [[4, 6]]):", dA)
    print("dB (expect [[16], [0]]):", dB.ravel())
