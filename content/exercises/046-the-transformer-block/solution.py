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
    z = np.asarray(z, dtype=np.float64)
    sigmoid = 0.5 * (1.0 + np.tanh(z / 2.0))  # equals 1 / (1 + e^-z) without overflow
    return z * sigmoid


def swiglu(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    gate = silu(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down


def ffn_parameter_count(d_model: int, d_ff: int, gated: bool) -> int:
    if not gated:
        return 2 * d_model * d_ff
    d_h = round(2 * d_ff / 3)
    return 3 * d_model * d_h


def layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gain + bias


def transformer_block_forward(x: np.ndarray, params: dict, mask: np.ndarray) -> np.ndarray:
    T, d_model = x.shape
    heads = params["heads"]
    if d_model % heads != 0:
        raise ValueError(f"heads={heads} does not divide d_model={d_model}")
    d_head = d_model // heads
    theta = params.get("theta", 10000.0)

    # Attention sublayer: read a normed copy, add the result to the raw stream.
    normed = layer_norm(x, params["ln1_gain"], params["ln1_bias"])

    def split_heads(t: np.ndarray) -> np.ndarray:
        return t.reshape(T, heads, d_head).transpose(1, 0, 2)  # (heads, T, d_head)

    q = split_heads(normed @ params["Wq"])
    k = split_heads(normed @ params["Wk"])
    v = split_heads(normed @ params["Wv"])

    cos, sin = rope_tables(T, d_head, theta)  # (T, d_head), broadcasts over heads
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head) + mask  # (heads, T, T)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    merged = (weights @ v).transpose(1, 0, 2).reshape(T, d_model)
    x1 = x + merged @ params["Wo"]

    # Feed-forward sublayer.
    normed2 = layer_norm(x1, params["ln2_gain"], params["ln2_bias"])
    return x1 + swiglu(normed2, params["W_gate"], params["W_up"], params["W_down"])


def block_parameter_count(d_model: int, d_ff: int, heads: int) -> int:
    if d_model % heads != 0:
        raise ValueError(f"heads={heads} does not divide d_model={d_model}")
    d_h = round(2 * d_ff / 3)
    return 4 * d_model**2 + 3 * d_model * d_h + 4 * d_model


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
