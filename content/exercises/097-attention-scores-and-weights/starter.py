"""Scores, softmax weights, and a weighted blend: the parts of a soft lookup.

Fill in the three functions below. Every array is one sequence with no batch
axis: queries are `(T_q, d_k)`, keys are `(T_k, d_k)`, and values are
`(T_k, d_v)`.
"""

import numpy as np


def dot_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Score every query against every key with a dot product.

    Do not divide by anything yet. That comes in the next chapter.

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


if __name__ == "__main__":
    # The word "it" asks about two candidates, "cat" and "sat".
    q = np.array([[1.0, 1.0]])
    k = np.array([[1.5, 0.5], [0.5, 0.5]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])

    scores = dot_scores(q, k)
    weights = softmax(scores, axis=-1)
    output = blend_values(weights, v)
    print("scores :", scores)
    print("weights:", weights.round(4), "sum", weights.sum())
    print("output :", output.round(4))
    print("softmax of [900, 0]:", softmax(np.array([900.0, 0.0])))
