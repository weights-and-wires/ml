"""A pre-norm transformer block: layer norm, rotary attention, SwiGLU, two residual adds.

Arrays are unbatched: a sequence is one (T, d_model) matrix.
"""

import numpy as np


# Provided: you wrote this in the chapter "Position encodings: sinusoidal clocks and rotary embeddings".
def rope_tables(T: int, d: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    speeds = theta ** (-2.0 * np.arange(d // 2) / d)
    angles = np.arange(T, dtype=np.float64)[:, None] * speeds[None, :]
    cos = np.concatenate([np.cos(angles)] * 2, axis=-1)
    sin = np.concatenate([np.sin(angles)] * 2, axis=-1)
    return cos, sin


# Provided: you wrote this in the chapter "Position encodings: sinusoidal clocks and rotary embeddings".
def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


def silu(z: np.ndarray) -> np.ndarray:
    """Return z * sigmoid(z), entry by entry.

    It must not overflow or warn for large negative z such as -800.

    Args:
        z: Array of any shape.

    Returns:
        Array with the same shape as z.
    """
    raise NotImplementedError("silu")


def swiglu(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    """Apply a SwiGLU feed-forward network to every row of x.

    swiglu(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    SiLU goes on the gate branch only.

    Args:
        x: Array of shape (..., d_model).
        W_gate: Array of shape (d_model, d_h).
        W_up: Array of shape (d_model, d_h).
        W_down: Array of shape (d_h, d_model).

    Returns:
        Array of shape (..., d_model).
    """
    raise NotImplementedError("swiglu")


def ffn_parameter_count(d_model: int, d_ff: int, gated: bool) -> int:
    """Return the number of weights in a feed-forward network, ignoring biases.

    A plain network has two matrices, d_model by d_ff and d_ff by d_model.
    A gated network has three matrices of the same size, with the hidden width
    d_h = round(2 * d_ff / 3) in place of d_ff.

    Args:
        d_model: Width of a token vector.
        d_ff: The feed-forward width of the plain network.
        gated: True for the SwiGLU network, False for the plain one.

    Returns:
        The weight count as an int.
    """
    raise NotImplementedError("ffn_parameter_count")


def layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalise each row of x to mean 0 and variance 1, then scale and shift.

    out = (x - mean) / sqrt(variance + eps) * gain + bias

    The mean and the variance are taken over the last axis, and the variance
    divides by d (numpy's default), not by d - 1.

    Args:
        x: Array of shape (..., d).
        gain: Array of shape (d,).
        bias: Array of shape (d,).
        eps: A small number added to the variance.

    Returns:
        Array with the same shape as x.
    """
    raise NotImplementedError("layer_norm")


def transformer_block_forward(x: np.ndarray, params: dict, mask: np.ndarray) -> np.ndarray:
    """Run one pre-norm transformer block.

    x1 = x + Attention(LayerNorm1(x))
    x2 = x1 + SwiGLU(LayerNorm2(x1))

    Attention projects the normed rows with Wq, Wk, and Wv, splits each result
    into params["heads"] heads of width d_model // heads, turns the queries and
    keys of each head with rotary embeddings (not the values), computes scores
    q @ k^T / sqrt(d_head), adds `mask`, takes a softmax over the keys,
    multiplies by the values, merges the heads, and multiplies by Wo.

    params holds:
        Wq, Wk, Wv, Wo: arrays of shape (d_model, d_model).
        ln1_gain, ln1_bias, ln2_gain, ln2_bias: arrays of shape (d_model,).
        W_gate, W_up: arrays of shape (d_model, d_h). W_down: shape (d_h, d_model).
        heads: int, must divide d_model.
        theta: float, the rotary base. Use 10000.0 if the key is absent.

    Args:
        x: Array of shape (T, d_model).
        params: Dict as described above.
        mask: Array of shape (T, T) added to the scores before the softmax.
            Use 0 to allow a position and -1e9 to forbid it.

    Returns:
        Array of shape (T, d_model).
    """
    raise NotImplementedError("transformer_block_forward")


def block_parameter_count(d_model: int, d_ff: int, heads: int) -> int:
    """Return the exact parameter count of one pre-norm block with SwiGLU.

    Attention holds four (d_model, d_model) matrices. SwiGLU holds three
    matrices with hidden width d_h = round(2 * d_ff / 3). The two layer norms
    hold a gain and a bias of length d_model each. Ignore other biases.

    Args:
        d_model: The model width.
        d_ff: The nominal feed-forward width of a plain network.
        heads: Number of attention heads. It must divide d_model.

    Returns:
        The parameter count as an int.

    Raises:
        ValueError: If heads does not divide d_model.
    """
    raise NotImplementedError("block_parameter_count")


def _demo_params(rng, d_model, d_ff, heads):
    d_h = round(2 * d_ff / 3)

    def small(*size):
        return rng.normal(size=size) * 0.1

    return {
        "Wq": small(d_model, d_model), "Wk": small(d_model, d_model),
        "Wv": small(d_model, d_model), "Wo": small(d_model, d_model),
        "ln1_gain": np.ones(d_model), "ln1_bias": np.zeros(d_model),
        "ln2_gain": np.ones(d_model), "ln2_bias": np.zeros(d_model),
        "W_gate": small(d_model, d_h), "W_up": small(d_model, d_h),
        "W_down": small(d_h, d_model), "heads": heads, "theta": 10000.0,
    }


if __name__ == "__main__":
    print("silu at -3, 0, 1, 3:", np.round(silu(np.array([-3.0, 0.0, 1.0, 3.0])), 4))

    x = np.array([1.0, 2.0])
    W_gate = np.array([[-3.0, 1.0], [0.0, 1.0]])
    W_up = np.eye(2)
    W_down = np.array([[1.0, 0.0], [1.0, -1.0]])
    print("swiglu on the hand example:", np.round(swiglu(x, W_gate, W_up, W_down), 3))
    print("plain weights at d=64, d_ff=256:", ffn_parameter_count(64, 256, gated=False))
    print("gated weights at d=64, d_ff=256:", ffn_parameter_count(64, 256, gated=True))

    row = np.array([[1.0, 2.0, 3.0, 6.0]])
    print("layer_norm of [1, 2, 3, 6]:", np.round(layer_norm(row, np.ones(4), np.zeros(4)), 4))

    rng = np.random.default_rng(0)
    T, d_model, d_ff, heads = 6, 16, 48, 4
    x = rng.normal(size=(T, d_model))
    params = _demo_params(rng, d_model, d_ff, heads)
    causal = np.triu(np.full((T, T), -1e9), k=1)
    out = transformer_block_forward(x, params, causal)
    print("block output shape:", out.shape, "(input", x.shape, ")")
    print("how far the block moved the stream:", round(float(np.abs(out - x).mean()), 4))
    print("block parameters at d=64, d_ff=192, 8 heads:", block_parameter_count(64, 192, 8))
