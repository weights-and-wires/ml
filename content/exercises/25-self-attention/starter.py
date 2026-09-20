"""Scaled dot-product and multi-head attention.

You implement the mechanics of self-attention: a stable softmax, single-head
scaled dot-product attention with an optional additive mask, multi-head
attention built on top of it, a causal mask, a flop counter, and the backward
pass for the single-head case.

A single sequence has no batch dimension here: `x` is `(T, d)`, and `q`, `k`,
`v` are `(Tq, d_k)`, `(Tk, d_k)`, `(Tk, d_v)` respectively, with `Tq == Tk` for
self-attention.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A numerically stable softmax along `axis`.

    Args:
        x: Input array, any shape.
        axis: Axis to normalise over.

    Returns:
        An array of the same shape as x, non-negative, summing to one along
        `axis`.
    """
    raise NotImplementedError("softmax")


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
):
    """Scaled dot-product attention for one or more queries.

    Args:
        q: Queries, shape (Tq, d_k).
        k: Keys, shape (Tk, d_k).
        v: Values, shape (Tk, d_v).
        mask: Optional additive mask, shape (Tq, Tk), added to the scores
            before the softmax. Use 0 to allow a position and -inf (or a very
            negative number) to forbid it.

    Returns:
        A tuple `(output, weights)`.
        output: Shape (Tq, d_v).
        weights: Attention weights, shape (Tq, Tk), each row non-negative and
            summing to one.
    """
    raise NotImplementedError("scaled_dot_product_attention")


def multi_head_attention(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-head self-attention over a single sequence.

    Args:
        x: Input sequence, shape (T, d).
        Wq, Wk, Wv, Wo: Projection matrices, each shape (d, d).
        heads: Number of heads. Must divide d evenly.
        mask: Optional additive mask, shape (T, T), shared across heads.

    Returns:
        The layer output, shape (T, d).
    """
    raise NotImplementedError("multi_head_attention")


def causal_mask(T: int) -> np.ndarray:
    """Return the additive causal mask for a sequence of length T.

    Args:
        T: Sequence length.

    Returns:
        An array of shape (T, T) with 0 at [t, j] for j <= t and -inf at
        [t, j] for j > t.
    """
    raise NotImplementedError("causal_mask")


def attention_flops(T: int, d: int, heads: int) -> int:
    """Return the flop count of one multi-head self-attention layer.

    Args:
        T: Sequence length.
        d: Model width.
        heads: Number of heads (accepted, but the total should not depend on
            it, as the lesson derives).

    Returns:
        The hand-computed flop count, 8*T*d**2 + 4*T**2*d.
    """
    raise NotImplementedError("attention_flops")


def attention_backward(
    dout: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
):
    """Backpropagate through single-head scaled_dot_product_attention.

    Recomputes the forward pass internally, then returns the gradient of
    `sum(dout * output)` with respect to q, k, and v.

    Args:
        dout: Upstream gradient on the output, shape (Tq, d_v).
        q, k, v, mask: The same arguments passed to
            `scaled_dot_product_attention`.

    Returns:
        A tuple `(dq, dk, dv)`, matching the shapes of q, k, and v.
    """
    raise NotImplementedError("attention_backward")


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
