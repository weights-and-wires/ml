"""Scaled dot-product attention, and the measurement that explains the scale.

Fill in the three functions below. `softmax` is provided: you wrote it in the
chapter "Letting tokens look at each other: scores and softmax weights".
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Written in the chapter "Letting tokens look at each other: scores and softmax weights".
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def attention_score_scale(d_k: int) -> float:
    """The factor that every raw score is multiplied by.

    Args:
        d_k: The width of each query and key vector.

    Returns:
        The number `1 / sqrt(d_k)` as a float. For `d_k = 64` it is `0.125`.
    """
    raise NotImplementedError("attention_score_scale")


def score_std(d_k: int, n: int = 2000, seed: int = 0) -> float:
    """Measure the spread of raw dot products between random vectors.

    Draw `n` queries and `n` keys, each of width `d_k`, with independent
    standard normal entries (mean 0, variance 1), from
    `np.random.default_rng(seed)`. Take the dot product of query `i` with key
    `i` for each `i`, and return the standard deviation of those `n` numbers.

    Args:
        d_k: The vector width.
        n: How many query-key pairs to sample.
        seed: The seed for the random generator.

    Returns:
        The standard deviation as a float. It is close to `sqrt(d_k)`.
    """
    raise NotImplementedError("score_std")


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
):
    """Scaled dot-product attention for one sequence.

    Compute `softmax(q @ k.T / sqrt(d_k)) @ v`, with `d_k = q.shape[-1]` and the
    softmax along each row. Leave `mask` as `None` in this chapter. A later
    chapter explains it; if it is given, add it to the scaled scores before
    the softmax.

    Args:
        q: Queries, shape `(T_q, d_k)`.
        k: Keys, shape `(T_k, d_k)`.
        v: Values, shape `(T_k, d_v)`.
        mask: Optional `(T_q, T_k)` array added to the scores.

    Returns:
        A pair `(output, weights)` with shapes `(T_q, d_v)` and `(T_q, T_k)`.
    """
    raise NotImplementedError("scaled_dot_product_attention")


if __name__ == "__main__":
    for d_k in (4, 16, 64, 256):
        raw = score_std(d_k)
        print(f"d_k={d_k:4d}  raw std {raw:6.2f}  after scaling {raw * attention_score_scale(d_k):5.2f}")

    # Two tokens with 64-wide vectors, each filled with one number.
    q = np.array([[0.25] * 64, [0.125] * 64])
    k = np.array([[1.0] * 64, [0.5] * 64])
    out, weights = scaled_dot_product_attention(q, k, np.eye(2))
    print("weights:\n", weights.round(4))
