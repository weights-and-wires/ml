"""Scores, softmax weights, a weighted blend, and the scale that keeps them soft.

Fill in the six functions below. Every array is one sequence with no batch
axis: queries are `(T_q, d_k)`, keys are `(T_k, d_k)`, and values are
`(T_k, d_v)`.
"""

import numpy as np


def dot_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Score every query against every key with a dot product.

    Do not divide by anything here. The scale is applied in
    `scaled_dot_product_attention`.

    Args:
        q: Queries, shape `(T_q, d_k)`.
        k: Keys, shape `(T_k, d_k)`.

    Returns:
        Scores of shape `(T_q, T_k)`, where entry `[i, j]` is the dot product
        of query `i` with key `j`.
    """
    raise NotImplementedError("dot_scores")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A numerically stable softmax along `axis`.

    Subtract the maximum along `axis` before exponentiating, so that a score
    such as 900 does not overflow.

    Args:
        x: Scores of any shape.
        axis: The axis whose entries become weights that sum to 1.

    Returns:
        An array of the same shape as `x`, positive, summing to 1 along `axis`.
    """
    raise NotImplementedError("softmax")


def blend_values(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Blend the value rows, each row of `weights` choosing the mix.

    Args:
        weights: Attention weights, shape `(T_q, T_k)`, each row summing to 1.
        v: Values, shape `(T_k, d_v)`.

    Returns:
        The blended values, shape `(T_q, d_v)`.
    """
    raise NotImplementedError("blend_values")


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
    softmax along each row. Leave `mask` as `None` here. A later chapter
    explains it; if it is given, add it to the scaled scores before the
    softmax.

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
    # The word "it" asks about two candidates, "cat" and "sat".
    q = np.array([[1.0, 1.0]])
    k = np.array([[1.5, 0.5], [0.5, 0.5]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = dot_scores(q, k)
    weights = softmax(scores, axis=-1)
    print("scores :", scores)
    print("weights:", weights.round(4), "sum", weights.sum())
    print("output :", blend_values(weights, v).round(4))
    print("softmax of [900, 0]:", softmax(np.array([900.0, 0.0])))

    for d_k in (4, 16, 64, 256):
        raw = score_std(d_k)
        print(f"d_k={d_k:4d}  raw std {raw:6.2f}  after scaling {raw * attention_score_scale(d_k):5.2f}")

    q = np.array([[0.25] * 64, [0.125] * 64])
    k = np.array([[1.0] * 64, [0.5] * 64])
    out, weights = scaled_dot_product_attention(q, k, np.eye(2))
    print("weights:\n", weights.round(4))
