"""Reference solution: scores, softmax weights, and a weighted blend."""

import numpy as np


def dot_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    return q @ k.T


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def blend_values(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    return weights @ v


if __name__ == "__main__":
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
