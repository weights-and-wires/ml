"""Scaled attention, a causal mask, and multi-head attention.

Shapes throughout: a sequence of `T` tokens has width `d`, so `x` is `(T, d)`.
Queries and keys share a width `d_k`, and values have width `d_v`.
"""

import numpy as np


def dot_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Dot product of every query with every key, with no scaling.

    Args:
        q: Queries, shape (T_q, d_k).
        k: Keys, shape (T_k, d_k).

    Returns:
        Scores, shape (T_q, T_k). Entry [i, j] is `q[i] @ k[j]`.
    """
    raise NotImplementedError("dot_scores")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along `axis`, stable for large scores.

    Subtract the maximum along `axis` before exponentiating, so a score of 900
    does not overflow. The result is the same as without the subtraction.

    Args:
        x: Scores of any shape.
        axis: The axis whose entries become weights that sum to one.

    Returns:
        Weights with the same shape as `x`.
    """
    raise NotImplementedError("softmax")


def blend_values(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Blend the value rows with each query's weights.

    Args:
        weights: Shape (T_q, T_k).
        v: Values, shape (T_k, d_v).

    Returns:
        Output, shape (T_q, d_v). Each row is a weighted sum of the rows of `v`.
    """
    raise NotImplementedError("blend_values")


def attention_score_scale(d_k: int) -> float:
    """The factor that brings the spread of a dot product back to about 1.

    Args:
        d_k: Width of the queries and keys.

    Returns:
        `1 / sqrt(d_k)` as a float.
    """
    raise NotImplementedError("attention_score_scale")


def score_std(d_k: int, n: int = 2000, seed: int = 0) -> float:
    """Measure the spread of `q @ k` for random vectors of width `d_k`.

    Draw `n` pairs of vectors with independent standard-normal entries, take
    the dot product of each pair, and return the standard deviation of those
    `n` numbers. Seed the generator with `np.random.default_rng(seed)`.

    Args:
        d_k: Width of each vector.
        n: Number of pairs.
        seed: Seed for the random generator.

    Returns:
        The standard deviation of the `n` dot products, as a float.
    """
    raise NotImplementedError("score_std")


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
):
    """Attention for one sequence: `softmax(q @ k.T / sqrt(d_k) + mask) @ v`.

    Args:
        q: Queries, shape (T_q, d_k).
        k: Keys, shape (T_k, d_k).
        v: Values, shape (T_k, d_v).
        mask: Optional array of shape (T_q, T_k) added to the scaled scores
            before the softmax. Use 0 where reading is allowed and -inf where
            it is not.

    Returns:
        A tuple `(output, weights)` with shapes (T_q, d_v) and (T_q, T_k).
        Each row of `weights` sums to one.
    """
    raise NotImplementedError("scaled_dot_product_attention")


def causal_mask(T: int) -> np.ndarray:
    """The additive mask that hides the future.

    Args:
        T: Number of tokens.

    Returns:
        Shape (T, T). Entry [i, j] is 0 when j <= i and `-np.inf` when j > i.
    """
    raise NotImplementedError("causal_mask")


def split_heads(x: np.ndarray, heads: int) -> list:
    """Cut the columns of `x` into `heads` contiguous blocks.

    Args:
        x: Shape (T, d).
        heads: Number of heads. It must divide `d`.

    Returns:
        A list of `heads` arrays, each of shape (T, d // heads). Head i holds
        columns `i * d_k` to `(i + 1) * d_k`.

    Raises:
        ValueError: If `heads` does not divide `d`.
    """
    raise NotImplementedError("split_heads")


def multi_head_attention(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-head attention on one sequence.

    Project `x` once with each of `Wq`, `Wk`, and `Wv`, split each projection
    into column blocks with `split_heads`, run `scaled_dot_product_attention`
    on each head with the same `mask`, concatenate the head outputs on the last
    axis, and multiply by `Wo`.

    Args:
        x: Shape (T, d).
        Wq, Wk, Wv, Wo: Weight matrices, each shape (d, d).
        heads: Number of heads. It must divide `d`.
        mask: Optional additive mask of shape (T, T), shared by every head.

    Returns:
        Output, shape (T, d).
    """
    raise NotImplementedError("multi_head_attention")


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

    print(causal_mask(4))
    x = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    eye = np.eye(4)
    print("two-head output:\n", multi_head_attention(x, eye, eye, eye, eye, heads=2).round(4))

    rng = np.random.default_rng(0)
    big = rng.normal(size=(6, 8))
    W = [0.3 * rng.normal(size=(8, 8)) for _ in range(4)]
    masked = multi_head_attention(big, *W, heads=4, mask=causal_mask(6))
    print("masked output shape:", masked.shape)

