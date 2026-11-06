"""Reference solution: causal mask, head splitting, and multi-head attention."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Written in the chapter "Letting tokens look at each other: scores and softmax weights".
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    # Written in the chapter "Why scores are divided by the square root of the width".
    scores = (q @ k.T) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


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
    print(causal_mask(4))

    x = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    eye = np.eye(4)
    out = multi_head_attention(x, eye, eye, eye, eye, heads=2)
    print("output:\n", out.round(4))

    rng = np.random.default_rng(0)
    big = rng.normal(size=(6, 8))
    W = [0.3 * rng.normal(size=(8, 8)) for _ in range(4)]
    masked = multi_head_attention(big, *W, heads=4, mask=causal_mask(6))
    print("masked output shape:", masked.shape)
