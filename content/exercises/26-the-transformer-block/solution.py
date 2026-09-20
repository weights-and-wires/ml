"""Reference solution for the transformer block exercise."""

import numpy as np


def sinusoidal_encoding(T: int, d: int) -> np.ndarray:
    positions = np.arange(T, dtype=np.float64)[:, None]        # (T, 1)
    i = np.arange(d // 2, dtype=np.float64)[None, :]           # (1, d/2)
    omega = 10000.0 ** (-2.0 * i / d)                           # (1, d/2)
    angles = positions * omega                                  # (T, d/2)

    encoding = np.empty((T, d), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding


def rope_frequencies(d: int, theta: float = 10000.0) -> np.ndarray:
    i = np.arange(d // 2, dtype=np.float64)
    return theta ** (-2.0 * i / d)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return x * cos + _rotate_half(x) * sin


def _rope_tables(T: int, d_head: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = rope_frequencies(d_head, theta)               # (d_head / 2,)
    positions = np.arange(T, dtype=np.float64)[:, None]     # (T, 1)
    angles = positions * freqs[None, :]                      # (T, d_head / 2)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (T, d_head)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos, sin


def swiglu(
    x: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray
) -> np.ndarray:
    gate = x @ W_gate
    silu = gate * (1.0 / (1.0 + np.exp(-gate)))  # silu(z) = z * sigmoid(z)
    up = x @ W_up
    return (silu * up) @ W_down


def _layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * gain + bias


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def transformer_block_forward(x: np.ndarray, params: dict, mask: np.ndarray) -> np.ndarray:
    T, d_model = x.shape
    heads = params["heads"]
    if d_model % heads != 0:
        raise ValueError(f"heads={heads} does not divide d_model={d_model}")
    d_head = d_model // heads
    theta = params.get("theta", 10000.0)

    # --- attention sublayer ---
    normed = _layer_norm(x, params["ln1_gain"], params["ln1_bias"])
    q = normed @ params["Wq"]
    k = normed @ params["Wk"]
    v = normed @ params["Wv"]

    def split_heads(t: np.ndarray) -> np.ndarray:
        return t.reshape(T, heads, d_head).transpose(1, 0, 2)  # (heads, T, d_head)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    cos, sin = _rope_tables(T, d_head, theta)  # (T, d_head), broadcasts over heads
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)  # (heads, T, T)
    scores = scores + mask
    weights = _softmax(scores)
    attn = weights @ v  # (heads, T, d_head)

    attn = attn.transpose(1, 0, 2).reshape(T, d_model)
    attn_out = attn @ params["Wo"]
    x1 = x + attn_out

    # --- feed-forward sublayer ---
    normed2 = _layer_norm(x1, params["ln2_gain"], params["ln2_bias"])
    ffn_out = swiglu(normed2, params["W_gate"], params["W_up"], params["W_down"])
    x2 = x1 + ffn_out
    return x2


def block_parameter_count(d_model: int, d_ff: int, heads: int) -> int:
    if d_model % heads != 0:
        raise ValueError(f"heads={heads} does not divide d_model={d_model}")
    attn_params = 4 * d_model**2
    d_h = round(2 * d_ff / 3)
    ffn_params = 3 * d_model * d_h
    norm_params = 2 * (2 * d_model)  # two layer norms, gain + bias each
    return attn_params + ffn_params + norm_params


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
