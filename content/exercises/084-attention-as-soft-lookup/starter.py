"""Attention as a soft lookup: softmax, the blend, and dot-product scores.

Shapes throughout: `query` is `(d_s,)`, `keys` is `(T, d_h)`, `values` is
`(T, d_v)`. One query looks at all `T` source positions at once.
"""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn scores into weights that are positive and sum to one.

    Subtract the largest score before exponentiating, so large scores do not
    overflow. The result is the same as without the subtraction.

    Args:
        z: Scores, shape (T,).

    Returns:
        Weights, shape (T,), each in [0, 1] and summing to 1.
    """
    raise NotImplementedError("softmax")


def attend(scores: np.ndarray, values: np.ndarray):
    """Blend the value rows, weighted by the softmax of the scores.

    Args:
        scores: One score per source position, shape (T,).
        values: The rows to blend, shape (T, d_v).

    Returns:
        A tuple `(context, weights)`.
        context: The weighted blend of the value rows, shape (d_v,).
        weights: The softmax of the scores, shape (T,).
    """
    raise NotImplementedError("attend")


def multiplicative_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W: np.ndarray | None = None,
):
    """Attention whose score is a dot product, with an optional matrix in the middle.

    With `W` equal to None, score i is `query @ keys[i]`, which needs
    `query.shape == (keys.shape[1],)`. With `W` given, score i is
    `query @ W @ keys[i]`, so the query and the keys may have different widths.

    Args:
        query: Shape (d_s,).
        keys: Shape (T, d_h).
        values: Shape (T, d_v).
        W: Shape (d_s, d_h), or None for the plain dot product.

    Returns:
        A tuple `(context, weights)` with shapes (d_v,) and (T,).
    """
    raise NotImplementedError("multiplicative_attention")


if __name__ == "__main__":
    query = np.array([1.0, 0.0])
    keys = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = np.array([[10.0, 0.0], [0.0, 10.0]])

    context, weights = multiplicative_attention(query, keys, values)
    print(f"weights {weights.round(4)}")
    print(f"context {context.round(3)}")

    sharp, sharp_weights = attend(np.array([10.0, 0.0]), values)
    print(f"scores [10, 0] give weights {sharp_weights.round(5)} and context {sharp.round(3)}")
    print(f"scores [101, 100] give weights {softmax(np.array([101.0, 100.0])).round(4)}")
