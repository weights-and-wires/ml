"""A pre-norm transformer block: attention, SwiGLU, and two ways to encode position.

Everything you need is built in this file; nothing is imported from the
self-attention exercise. Work bottom-up: the position-encoding functions and
`swiglu` stand alone, and `transformer_block_forward` wires them together with
an attention computation you write inline.

Arrays are unbatched: a sequence is a single `(T, d_model)` matrix, not a
`(batch, T, d_model)` tensor.
"""

import numpy as np


def sinusoidal_encoding(T: int, d: int) -> np.ndarray:
    """Return the fixed sinusoidal position encoding matrix.

    Row p, column 2i holds sin(p * omega_i); column 2i + 1 holds
    cos(p * omega_i), for omega_i = 10000 ** (-2i / d) and i = 0, ..., d/2 - 1.
    d must be even.

    Args:
        T: Number of positions.
        d: Encoding dimension, even.

    Returns:
        Array of shape (T, d).
    """
    raise NotImplementedError("sinusoidal_encoding")


def rope_frequencies(d: int, theta: float = 10000.0) -> np.ndarray:
    """Return the rotary inverse frequencies for a head dimension d.

    Frequency i is theta ** (-2i / d), for i = 0, ..., d/2 - 1. d must be even.

    Args:
        d: Head dimension, even.
        theta: The base of the geometric progression.

    Returns:
        Array of shape (d // 2,).
    """
    raise NotImplementedError("rope_frequencies")


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply rotary position embeddings to x, using the rotate-half convention.

    Split the last axis of x into two equal halves x1, x2. The rotated result
    is x1, x2 each rotated by a per-position, per-frequency angle:

        out = x * cos + rotate_half(x) * sin
        rotate_half(x) = concatenate([-x2, x1], axis=-1)

    cos and sin have the frequency for index i repeated at index i + d/2, so
    they broadcast against x's last axis directly. They broadcast against any
    leading shape of x (a single vector, a (T, d) sequence, or a
    (heads, T, d) stack of heads).

    Args:
        x: Array whose last axis has even size d.
        cos: Array of cosines, broadcastable against x, last axis size d.
        sin: Array of sines, broadcastable against x, last axis size d.

    Returns:
        Array the same shape as x.
    """
    raise NotImplementedError("apply_rope")


def swiglu(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    """Apply a SwiGLU feed-forward network to every row of x.

    SwiGLU(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down, with
    silu(z) = z * sigmoid(z).

    Args:
        x: Array of shape (..., d_model).
        W_gate: Array of shape (d_model, d_h).
        W_up: Array of shape (d_model, d_h).
        W_down: Array of shape (d_h, d_model).

    Returns:
        Array of shape (..., d_model).
    """
    raise NotImplementedError("swiglu")


def transformer_block_forward(x: np.ndarray, params: dict, mask: np.ndarray) -> np.ndarray:
    """Run one pre-norm transformer block: attention, then a SwiGLU network.

    x' = x + Attention(LayerNorm(x))
    x'' = x' + SwiGLU(LayerNorm(x'))

    Attention splits its queries and keys into params["heads"] heads of size
    d_model // heads, applies rotary position embeddings to each head's
    queries and keys before the scaled dot product, adds `mask` to the raw
    scores, and softmaxes over the key axis. The heads are concatenated back
    to d_model and passed through the output projection Wo.

    params holds:
        Wq, Wk, Wv, Wo: arrays of shape (d_model, d_model).
        ln1_gain, ln1_bias, ln2_gain, ln2_bias: arrays of shape (d_model,).
        W_gate, W_up, W_down: shapes (d_model, d_h), (d_model, d_h), (d_h, d_model).
        heads: int, must divide d_model evenly.
        theta: float, the RoPE base (default 10000.0 if absent).

    Args:
        x: Array of shape (T, d_model).
        params: Dict of arrays and scalars as described above.
        mask: Array of shape (T, T), added to the raw attention scores before
            the softmax. Use 0 to allow a position and a large negative number
            (or -inf) to forbid it.

    Returns:
        Array of shape (T, d_model).
    """
    raise NotImplementedError("transformer_block_forward")


def block_parameter_count(d_model: int, d_ff: int, heads: int) -> int:
    """Return the exact parameter count of one pre-norm block with SwiGLU.

    The feed-forward network is SwiGLU with its hidden width scaled to
    two-thirds of d_ff, so its three matrices cost as many parameters as a
    plain two-matrix network of hidden size d_ff would. Include both layer
    norms' gain and bias vectors. heads must divide d_model evenly; it does
    not otherwise change the count, since redistributing d_model among more,
    narrower heads does not add parameters.

    Args:
        d_model: The model width.
        d_ff: The nominal (ungated) feed-forward hidden size.
        heads: Number of attention heads. Must divide d_model.

    Returns:
        The total parameter count as an int.
    """
    raise NotImplementedError("block_parameter_count")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d_model, d_ff, heads = 12, 32, 128, 4
    d_h = round(2 * d_ff / 3)

    x = rng.normal(size=(T, d_model)) * 0.1
    params = {
        "Wq": rng.normal(size=(d_model, d_model)) * 0.02,
        "Wk": rng.normal(size=(d_model, d_model)) * 0.02,
        "Wv": rng.normal(size=(d_model, d_model)) * 0.02,
        "Wo": rng.normal(size=(d_model, d_model)) * 0.02,
        "ln1_gain": np.ones(d_model),
        "ln1_bias": np.zeros(d_model),
        "ln2_gain": np.ones(d_model),
        "ln2_bias": np.zeros(d_model),
        "W_gate": rng.normal(size=(d_model, d_h)) * 0.02,
        "W_up": rng.normal(size=(d_model, d_h)) * 0.02,
        "W_down": rng.normal(size=(d_h, d_model)) * 0.02,
        "heads": heads,
        "theta": 10000.0,
    }
    causal = np.triu(np.full((T, T), -1e9), k=1)

    out = transformer_block_forward(x, params, causal)
    print(f"input shape:  {x.shape}")
    print(f"output shape: {out.shape}")
    print(f"block parameters: {block_parameter_count(d_model, d_ff, heads):,}")
