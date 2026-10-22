"""Reference solution: softmax, the blend, and dot-product scores."""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    exp = np.exp(z - z.max())
    return exp / exp.sum()


def attend(scores: np.ndarray, values: np.ndarray):
    weights = softmax(scores)  # (T,)
    context = weights @ values  # (d_v,)
    return context, weights


def multiplicative_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W: np.ndarray | None = None,
):
    if W is None:
        scores = keys @ query  # (T,)
    else:
        scores = keys @ (W.T @ query)  # (T,): query @ W @ keys[i] for every i
    return attend(scores, values)


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
