"""Reference solution: scaled attention, causal mask, and multi-head attention."""

import numpy as np


def dot_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    return q @ k.T


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def blend_values(weights: np.ndarray, v: np.ndarray) -> np.ndarray:
    return weights @ v


def attention_score_scale(d_k: int) -> float:
    return float(1.0 / np.sqrt(d_k))


def score_std(d_k: int, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, d_k))
    k = rng.normal(size=(n, d_k))
    return float(np.std(np.sum(q * k, axis=1)))


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
):
    scores = dot_scores(q, k) * attention_score_scale(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return blend_values(weights, v), weights


def causal_mask(T: int) -> np.ndarray:
    mask = np.zeros((T, T))
    mask[np.triu_indices(T, k=1)] = -np.inf
    return mask


def split_heads(x: np.ndarray, heads: int) -> list:
    d = x.shape[-1]
    if d % heads != 0:
        raise ValueError(f"heads={heads} must divide d={d} evenly")
    d_k = d // heads
    return [x[:, i * d_k:(i + 1) * d_k] for i in range(heads)]


def multi_head_attention(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    qs = split_heads(x @ Wq, heads)
    ks = split_heads(x @ Wk, heads)
    vs = split_heads(x @ Wv, heads)
    outputs = [
        scaled_dot_product_attention(q, k, v, mask)[0] for q, k, v in zip(qs, ks, vs)
    ]
    return np.concatenate(outputs, axis=-1) @ Wo


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
