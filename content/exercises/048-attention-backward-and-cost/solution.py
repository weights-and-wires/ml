"""Reference solution: the gradient of attention and its flop count."""

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


def softmax_backward(weights: np.ndarray, dweights: np.ndarray) -> np.ndarray:
    rowsum = np.sum(weights * dweights, axis=-1, keepdims=True)
    return weights * (dweights - rowsum)


def attention_backward(
    dout: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = (q @ k.T) * scale
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)

    dv = weights.T @ dout                      # (T_k, d_v)
    dweights = dout @ v.T                      # (T_q, T_k)
    dscores = softmax_backward(weights, dweights)
    dq = (dscores @ k) * scale                 # (T_q, d_k)
    dk = (dscores.T @ q) * scale               # (T_k, d_k)
    return dq, dk, dv


def attention_flops(T: int, d: int, heads: int) -> int:
    projections = 4 * (2 * T * d * d)          # W_Q, W_K, W_V, W_O
    table = 2 * (2 * T * T * d)                # q @ k.T and weights @ v
    return int(projections + table)


if __name__ == "__main__":
    q = np.ones((1, 4))
    k = np.array([[1.0] * 4, [0.5] * 4])
    v = np.eye(2)
    dout = np.array([[1.0, -1.0]])

    out, weights = scaled_dot_product_attention(q, k, v)
    print("weights:", weights.round(4))
    print("softmax_backward:", softmax_backward(weights, dout @ v.T).round(4))
    dq, dk, dv = attention_backward(dout, q, k, v)
    print("dq:", dq.round(4))
    print("dk:\n", dk.round(4))
    print("dv:\n", dv.round(4))

    print("flops for T=16, d=4:", attention_flops(16, 4, 1))
    for T in (64, 128, 256):
        print(f"T={T:4d}: projections {8 * T * 64**2:>10,d}  score table {4 * T**2 * 64:>10,d}")
