"""Online softmax and tiled attention.

Every array is one head with no batch axis: `q` is `(T_q, d)`, `k` is
`(T_k, d)`, and `v` is `(T_k, d_v)`.
"""

from __future__ import annotations

import math

import numpy as np


def dense_attention(q, k, v, causal=False):
    """Dense reference, provided. Builds the whole `(T_q, T_k)` score table."""
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    if causal:
        scores = np.where(np.tril(np.ones(scores.shape, dtype=bool)), scores, -np.inf)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def block_state(scores, v):
    """Provided. Summarise one block of scores as a state `(m, l, o)`.

    `scores` is `(n, B)` and `v` is `(B, d_v)`. Returns the row max `m` with
    shape `(n,)`, the sum `l` of `exp(scores - m)` with shape `(n,)`, and the
    unnormalised output `o = exp(scores - m) @ v` with shape `(n, d_v)`.
    """
    m = scores.max(axis=-1)
    e = np.exp(scores - m[:, None])
    return m, e.sum(axis=-1), e @ v


def online_stats(stream):
    """Return `(m, l)` for an iterable of scores, reading each score once."""
    m = -math.inf
    l = 0.0
    for x in stream:
        m_new = max(m, x)
        l = l * math.exp(m - m_new) + math.exp(x - m_new)
        m = m_new
    return m, l


def online_softmax(x):
    """Softmax of a 1-D array, with the max and the sum found in one pass."""
    x = np.asarray(x, dtype=float)
    m, l = online_stats(x)
    return np.exp(x - m) / l


def merge_states(a, b):
    """Merge two partial softmax states over disjoint sets of keys."""
    ma, la, oa = a
    mb, lb, ob = b
    m = np.maximum(ma, mb)
    alpha = np.exp(ma - m)
    beta = np.exp(mb - m)
    return m, la * alpha + lb * beta, oa * alpha[:, None] + ob * beta[:, None]


def count_causal_blocks(T, block):
    """Return `(computed, skipped)` tile counts for a causal table."""
    nb = -(-T // block)
    computed = nb * (nb + 1) // 2
    return computed, nb * nb - computed


def tiled_attention(q, k, v, block, causal=False):
    """Attention over `block` by `block` tiles. Returns `(out, lse, peak)`."""
    Tq, Tk = q.shape[0], k.shape[0]
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = np.empty((Tq, v.shape[-1]))
    lse = np.empty(Tq)
    peak = 0
    for i in range(0, Tq, block):
        qi = q[i : i + block]
        state = (
            np.full(qi.shape[0], -np.inf),
            np.zeros(qi.shape[0]),
            np.zeros((qi.shape[0], v.shape[-1])),
        )
        for j in range(0, Tk, block):
            if causal and j > i + qi.shape[0] - 1:
                break
            s = (qi @ k[j : j + block].T) * scale
            if causal:
                rows = np.arange(i, i + qi.shape[0])[:, None]
                cols = np.arange(j, j + s.shape[1])[None, :]
                s = np.where(cols <= rows, s, -np.inf)
            peak = max(peak, s.size)
            state = merge_states(state, block_state(s, v[j : j + block]))
        m, l, o = state
        out[i : i + block] = o / l[:, None]
        lse[i : i + block] = m + np.log(l)
    return out, lse, peak


def recompute_probs(q, k, lse, causal=False):
    """Rebuild the attention weights from `lse`, the log-sum-exp of each row."""
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    p = np.exp(scores - lse[:, None])
    if causal:
        p = np.where(np.tril(np.ones(p.shape, dtype=bool)), p, 0.0)
    return p


def backward_from_lse(dout, q, k, v, out, lse, causal=False):
    """Gradients `(dq, dk, dv)` using only `out` and `lse`, never a saved table."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    p = recompute_probs(q, k, lse, causal)
    big_d = np.sum(dout * out, axis=-1, keepdims=True)
    dv = p.T @ dout
    dp = dout @ v.T
    ds = p * (dp - big_d)
    return (ds @ k) * scale, (ds.T @ q) * scale, dv


def extra_saved(T, tiled):
    """Numbers saved per head for the backward pass, beyond q, k, v, and out."""
    return T if tiled else 2 * T * T


def score_table_flops(T, d, mode):
    """Flops of the score-table products for one layer of width `d`."""
    if mode == "forward":
        return 4 * T * T * d
    if mode == "backward":
        return 8 * T * T * d
    if mode == "backward_recompute":
        return 10 * T * T * d
    raise ValueError(f"unknown mode {mode!r}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(10, 4)) for _ in range(3))
    out, lse, peak = tiled_attention(q, k, v, block=4, causal=True)
    print("max error against dense:", np.abs(out - dense_attention(q, k, v, True)).max())
    print("score entries held at once:", peak)
