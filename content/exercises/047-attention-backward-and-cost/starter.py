"""The gradient of attention, and what one attention layer costs.

Fill in the three functions below. `softmax` and `scaled_dot_product_attention`
are provided: you wrote them in earlier chapters. Every array is one sequence
with no batch axis.
"""

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
    """Send a gradient back through a row-wise softmax.

    Args:
        weights: The softmax output, shape `(T_q, T_k)`, each row summing to 1.
        dweights: The gradient of the loss with respect to `weights`, same shape.

    Returns:
        The gradient of the loss with respect to the scores that went into the
        softmax, same shape. Each row of the result sums to 0.
    """
    raise NotImplementedError("softmax_backward")


def attention_backward(
    dout: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
):
    """Gradients of scaled dot-product attention.

    Recompute the forward pass, then undo its three steps in reverse: the
    blend `weights @ v`, the softmax, and the scaled product `q @ k.T / sqrt(d_k)`.

    Args:
        dout: The gradient with respect to the output, shape `(T_q, d_v)`.
        q: Queries `(T_q, d_k)`.
        k: Keys `(T_k, d_k)`.
        v: Values `(T_k, d_v)`.
        mask: Optional `(T_q, T_k)` additive mask used in the forward pass.

    Returns:
        A tuple `(dq, dk, dv)` with the same shapes as `q`, `k`, and `v`.
    """
    raise NotImplementedError("attention_backward")


def attention_flops(T: int, d: int, heads: int) -> int:
    """Count the flops of one multi-head attention layer's forward pass.

    One flop is one multiply or one add, so a `(p, q)` by `(q, r)` matrix
    product costs `2 * p * q * r`. Count the four `d x d` projections
    (`W_Q`, `W_K`, `W_V`, `W_O`) and the two `T x T` products (scores, and the
    blend with the values). The number of heads does not change the total.

    Args:
        T: Sequence length.
        d: Model width.
        heads: Number of heads, accepted and unused.

    Returns:
        The flop count as an `int`.
    """
    raise NotImplementedError("attention_flops")


if __name__ == "__main__":
    # The worked example: one query, two keys, identity values.
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
