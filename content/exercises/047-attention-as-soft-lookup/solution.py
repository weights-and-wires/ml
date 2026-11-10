"""Reference solution: soft lookup with dot-product and additive scores.

Shapes throughout: `query` is `(d_s,)`, `keys` is `(T, d_h)`, `values` is
`(T, d_v)`, `W1` is `(h, d_s)`, `W2` is `(h, d_h)`, and `v` is `(h,)`.
"""

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


def additive_scores(
    query: np.ndarray,
    keys: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    u = W1 @ query + keys @ W2.T  # (T, h): the query part is added to every row
    return np.tanh(u) @ v  # (T,)


def additive_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    v: np.ndarray,
):
    scores = additive_scores(query, keys, W1, W2, v)
    return attend(scores, values)


def emulate_dot_params(query: np.ndarray, c: float):
    d = query.shape[0]
    return c * np.eye(d), c * np.eye(d), query / c


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
