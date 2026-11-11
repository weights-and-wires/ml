"""Online softmax and tiled attention.

Fill in the eight functions marked `NotImplementedError`. `dense_attention` and
`block_state` are provided. Every array is one head with no batch axis: `q` is
`(T_q, d)`, `k` is `(T_k, d)`, and `v` is `(T_k, d_v)`.
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
    """Return `(m, l)` for an iterable of scores, reading each score once.

    Args:
        stream: Any iterable of floats. It may be a one-shot generator, so you
            can't index it or loop over it twice.

    Returns:
        The running max `m` and `l`, the sum of `exp(x - m)` over every score.
    """
    raise NotImplementedError("write online_stats")


def online_softmax(x):
    """Softmax of a 1-D array, with the max and the sum found in one pass.

    Args:
        x: A 1-D float array.

    Returns:
        An array of the same shape that sums to 1.
    """
    raise NotImplementedError("write online_softmax")


def merge_states(a, b):
    """Merge two partial softmax states over disjoint sets of keys.

    Args:
        a, b: States `(m, l, o)` with shapes `(n,)`, `(n,)`, and `(n, d_v)`,
            each relative to its own max.

    Returns:
        The state of the union, relative to the larger of the two maxima.
    """
    raise NotImplementedError("write merge_states")


def count_causal_blocks(T, block):
    """Return `(computed, skipped)` tile counts for a causal table.

    The table is cut into `ceil(T / block)` rows and columns of tiles. A tile
    that lies wholly above the diagonal is skipped.
    """
    raise NotImplementedError("write count_causal_blocks")


def tiled_attention(q, k, v, block, causal=False):
    """Attention that never builds a table larger than `block` by `block`.

    Args:
        q: `(T_q, d)`. k: `(T_k, d)`. v: `(T_k, d_v)`.
        block: Tile edge, for rows and for columns.
        causal: When True, query `i` reads only keys `0..i`, and tiles wholly
            above the diagonal are skipped.

    Returns:
        `(out, lse, peak)`: the output `(T_q, d_v)`, the log-sum-exp of every
        scaled score row `(T_q,)`, and the largest number of score entries
        held at once (the size of the largest tile).
    """
    raise NotImplementedError("write tiled_attention")


def recompute_probs(q, k, lse, causal=False):
    """Rebuild the attention weights `(T_q, T_k)` from `lse`, with no softmax.

    Args:
        lse: The log-sum-exp of each scaled score row, shape `(T_q,)`.
    """
    raise NotImplementedError("write recompute_probs")


def backward_from_lse(dout, q, k, v, out, lse, causal=False):
    """Gradients `(dq, dk, dv)` using only `out` and `lse`, never a saved table.

    Rebuild the weights with `recompute_probs`. The row correction needed by the
    softmax rule is `sum(dout * out, axis=-1)`.
    """
    raise NotImplementedError("write backward_from_lse")


def extra_saved(T, tiled):
    """Numbers saved per head for the backward pass, beyond q, k, v, and out.

    The dense layer saves the scores and the weights. The tiled layer saves
    only `lse`.
    """
    raise NotImplementedError("write extra_saved")


def score_table_flops(T, d, mode):
    """Flops of the score-table products for one layer of width `d`.

    `mode` is "forward", "backward" (weights were saved), or
    "backward_recompute" (the scores are rebuilt first).
    """
    raise NotImplementedError("write score_table_flops")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(10, 4)) for _ in range(3))
    out, lse, peak = tiled_attention(q, k, v, block=4, causal=True)
    print("max error against dense:", np.abs(out - dense_attention(q, k, v, True)).max())
    print("score entries held at once:", peak)
