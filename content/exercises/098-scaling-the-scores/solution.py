"""Reference solution: scaled dot-product attention and the spread of raw scores."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Written in the chapter "Letting tokens look at each other: scores and softmax weights".
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


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
    scores = (q @ k.T) * attention_score_scale(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


if __name__ == "__main__":
    for d_k in (4, 16, 64, 256):
        raw = score_std(d_k)
        print(f"d_k={d_k:4d}  raw std {raw:6.2f}  after scaling {raw * attention_score_scale(d_k):5.2f}")

    q = np.array([[0.25] * 64, [0.125] * 64])
    k = np.array([[1.0] * 64, [0.5] * 64])
    out, weights = scaled_dot_product_attention(q, k, np.eye(2))
    print("weights:\n", weights.round(4))
