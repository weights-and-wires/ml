"""Attention as a soft lookup: softmax, the blend, and two ways to score a match.

Shapes throughout: `query` is `(d_s,)`, `keys` is `(T, d_h)`, `values` is
`(T, d_v)`, `W1` is `(h, d_s)`, `W2` is `(h, d_h)`, and `v` is `(h,)`, where
`h` is the width of the additive scorer's hidden layer. One query looks at all
`T` source positions at once.
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


def additive_scores(
    query: np.ndarray,
    keys: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Score every key with `v @ tanh(W1 @ query + W2 @ keys[i])`.

    Args:
        query: Shape (d_s,).
        keys: Shape (T, d_h).
        W1: Query projection, shape (h, d_s).
        W2: Key projection, shape (h, d_h).
        v: Scoring vector, shape (h,).

    Returns:
        One score per key, shape (T,).
    """
    raise NotImplementedError("additive_scores")


def additive_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
):
    """Attention whose scores come from `additive_scores`.

    Args:
        query: Shape (d_s,).
        keys: Shape (T, d_h).
        values: Shape (T, d_v).
        W1, W2, v: The scorer's parameters, as in `additive_scores`.

    Returns:
        A tuple `(context, weights)` with shapes (d_v,) and (T,).
    """
    raise NotImplementedError("additive_attention")


def emulate_dot_params(query: np.ndarray, c: float):
    """Parameters that make additive attention imitate a plain dot product.

    Because tanh(c * x) / c is close to x when c is small, the choice
    `W1 = W2 = c * I` and `v = query / c` gives scores close to
    `query @ query + query @ keys[i]`. The first term is the same for every
    key and cancels in the softmax. The parameters are built from this one
    query, so they imitate the dot product for it only.

    Args:
        query: Shape (d,). The query and the keys share this width.
        c: A small positive number.

    Returns:
        A tuple `(W1, W2, v)` with shapes (d, d), (d, d), and (d,).
    """
    raise NotImplementedError("emulate_dot_params")


if __name__ == "__main__":
    query = np.array([1.0, 0.0])
    keys = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = np.array([[10.0, 0.0], [0.0, 10.0]])
    eye = np.eye(2)

    context, weights = multiplicative_attention(query, keys, values)
    print(f"dot product: weights {weights.round(4)}, context {context.round(3)}")
    print(f"scores [101, 100] give weights {softmax(np.array([101.0, 100.0])).round(4)}")

    v = np.array([1.0, 1.0])
    scores = additive_scores(query, keys, eye, eye, v)
    print(f"additive scores {scores.round(4)}")
    context, weights = additive_attention(query, keys, values, eye, eye, v)
    print(f"additive: weights {weights.round(4)}, context {context.round(3)}")
    _, flipped = additive_attention(query, keys, values, eye, eye, -v)
    print(f"with v = [-1, -1] the weights are {flipped.round(4)}")

    rng = np.random.default_rng(0)
    q = rng.normal(size=3)
    k = rng.normal(size=(5, 3))
    x = rng.normal(size=(5, 2))
    W1, W2, v = emulate_dot_params(q, 1e-3)
    _, additive_w = additive_attention(q, k, x, W1, W2, v)
    _, dot_w = multiplicative_attention(q, k, x)
    print(f"largest gap to a plain dot product {np.abs(additive_w - dot_w).max():.2e}")
