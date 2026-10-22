"""A causal mask, head splitting, and multi-head attention.

Fill in the three functions below. `softmax` and `scaled_dot_product_attention`
are provided: you wrote them in the two previous chapters. Every array is one
sequence with no batch axis, so `x` is `(T, d)`.
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


def causal_mask(T: int) -> np.ndarray:
    """The additive mask that hides future positions.

    Returns:
        A `(T, T)` float array that is `0.0` on and below the diagonal and
        `-np.inf` strictly above it. Row `t` is the reading position and
        column `j` is the position being read, so position `t` may read
        positions `0..t`.
    """
    raise NotImplementedError("causal_mask")


def split_heads(x: np.ndarray, heads: int) -> list:
    """Cut the columns of `x` into `heads` equal blocks.

    Args:
        x: An array of shape `(T, d)`.
        heads: The number of heads. It must divide `d`.

    Returns:
        A list of `heads` arrays, each of shape `(T, d // heads)`. Head `i`
        holds columns `i * (d // heads)` up to `(i + 1) * (d // heads)`.

    Raises:
        ValueError: If `heads` does not divide `d`.
    """
    raise NotImplementedError("split_heads")


def multi_head_attention(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-head self-attention.

    Project `x` with `Wq`, `Wk`, and `Wv` (each `(d, d)`), split each
    projection into `heads` column blocks, run scaled dot-product attention on
    each head with the same `mask`, concatenate the head outputs along the last
    axis, and multiply by `Wo`.

    Args:
        x: Token vectors, shape `(T, d)`.
        Wq, Wk, Wv, Wo: Weight matrices, each `(d, d)`.
        heads: The number of heads.
        mask: Optional `(T, T)` additive mask.

    Returns:
        An array of shape `(T, d)`.
    """
    raise NotImplementedError("multi_head_attention")


if __name__ == "__main__":
    print(causal_mask(4))

    # The worked example: two tokens, d = 4, two heads, identity weights.
    x = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    eye = np.eye(4)
    out = multi_head_attention(x, eye, eye, eye, eye, heads=2)
    print("output:\n", out.round(4))

    rng = np.random.default_rng(0)
    big = rng.normal(size=(6, 8))
    W = [0.3 * rng.normal(size=(8, 8)) for _ in range(4)]
    masked = multi_head_attention(big, *W, heads=4, mask=causal_mask(6))
    print("masked output shape:", masked.shape)
