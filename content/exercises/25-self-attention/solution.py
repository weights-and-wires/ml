"""Reference solution for the self-attention exercise."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
):
    d_k = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    output = weights @ v
    return output, weights


def multi_head_attention(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    d = x.shape[-1]
    if d % heads != 0:
        raise ValueError(f"heads={heads} must divide d={d} evenly")
    d_k = d // heads

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    head_outputs = []
    for i in range(heads):
        lo, hi = i * d_k, (i + 1) * d_k
        out_i, _ = scaled_dot_product_attention(Q[:, lo:hi], K[:, lo:hi], V[:, lo:hi], mask)
        head_outputs.append(out_i)

    concatenated = np.concatenate(head_outputs, axis=-1)
    return concatenated @ Wo


def causal_mask(T: int) -> np.ndarray:
    mask = np.zeros((T, T))
    mask[np.triu_indices(T, k=1)] = -np.inf
    return mask


def attention_flops(T: int, d: int, heads: int) -> int:
    return 8 * T * d ** 2 + 4 * T ** 2 * d


def attention_backward(
    dout: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
):
    d_k = q.shape[-1]
    scale = np.sqrt(d_k)
    scores = (q @ k.T) / scale
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)

    dweights = dout @ v.T  # (Tq, Tk)
    dv = weights.T @ dout  # (Tk, d_v)

    row_dot = np.sum(weights * dweights, axis=-1, keepdims=True)
    dscores = weights * (dweights - row_dot)  # (Tq, Tk)

    dq = (dscores @ k) / scale  # (Tq, d_k)
    dk = (dscores.T @ q) / scale  # (Tk, d_k)

    return dq, dk, dv


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d, heads = 6, 8, 2

    x = rng.normal(size=(T, d))
    Wq = 0.3 * rng.normal(size=(d, d))
    Wk = 0.3 * rng.normal(size=(d, d))
    Wv = 0.3 * rng.normal(size=(d, d))
    Wo = 0.3 * rng.normal(size=(d, d))

    mask = causal_mask(T)
    out = multi_head_attention(x, Wq, Wk, Wv, Wo, heads, mask)
    print(f"multi-head output shape {out.shape}")

    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    single_out, weights = scaled_dot_product_attention(q, k, v, mask)
    print(f"single-head weights row sums: {weights.sum(axis=-1).round(6)}")

    dout = rng.normal(size=single_out.shape) * 0.1
    dq, dk, dv = attention_backward(dout, q, k, v, mask)
    print(f"dq shape {dq.shape}, dk shape {dk.shape}, dv shape {dv.shape}")

    for h in (1, 2, 4, 8):
        print(f"heads={h}: {attention_flops(T, d, h)} flops")
