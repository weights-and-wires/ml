"""Modern transformer choices: RMSNorm, grouped-query attention, parameter counts, and norm placement.

Arrays are unbatched unless a docstring says otherwise.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-5


def rmsnorm(x: np.ndarray, gain: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Scale each row of x to unit root mean square, then apply the gain.

    y = x / sqrt(mean(x**2) + eps) * gain, with the mean over the last axis.
    Unlike layer norm, nothing is subtracted.

    Args:
        x: Array of shape (..., d).
        gain: Array of shape (d,).
        eps: Small constant added to the mean square.

    Returns:
        Array with the same shape as x.
    """
    inv_rms = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * inv_rms * gain


def rmsnorm_backward(
    dy: np.ndarray, x: np.ndarray, gain: np.ndarray, eps: float = EPS
) -> tuple[np.ndarray, np.ndarray]:
    """Backpropagate through rmsnorm.

    Args:
        dy: Gradient of the loss with respect to the output, shape (..., d).
        x: The input that rmsnorm received, shape (..., d).
        gain: Array of shape (d,).
        eps: The same constant that the forward pass used.

    Returns:
        (dx, dgain). dx has the shape of x. dgain has shape (d,) and sums over
        every leading axis.
    """
    inv_rms = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    xhat = x * inv_rms
    u = dy * gain
    dx = inv_rms * (u - xhat * np.mean(u * xhat, axis=-1, keepdims=True))
    dgain = (dy * xhat).reshape(-1, x.shape[-1]).sum(axis=0)
    return dx, dgain


def gqa_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, n_groups: int, causal: bool = True
) -> np.ndarray:
    """Grouped-query attention.

    Query head h reads key-value group h // (H // n_groups), so the first H / n_groups
    query heads share group 0, the next H / n_groups share group 1, and so on.

    Args:
        q: Queries, shape (H, T, d_head).
        k: Keys, shape (n_groups, T, d_head).
        v: Values, shape (n_groups, T, d_head).
        n_groups: Number of key-value heads. Must divide H and equal k.shape[0].
        causal: When True, position t attends only to positions 0 to t.

    Returns:
        Array of shape (H, T, d_head).
    """
    H, T, d_head = q.shape
    if n_groups < 1 or H % n_groups != 0 or k.shape[0] != n_groups or v.shape[0] != n_groups:
        raise ValueError(f"n_groups={n_groups} must divide H={H} and match k and v {k.shape[0]}, {v.shape[0]}")
    per_group = H // n_groups
    k_full = np.repeat(k, per_group, axis=0)
    v_full = np.repeat(v, per_group, axis=0)
    scores = q @ k_full.transpose(0, 2, 1) / np.sqrt(d_head)
    if causal:
        scores = scores + np.triu(np.full((T, T), -1e9), k=1)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v_full


def kv_cache_bytes(
    n_layers: int, n_kv_heads: int, d_head: int, seq_len: int, bytes_per_value: int = 2
) -> int:
    """Bytes that the keys and values of one sequence occupy.

    Args:
        n_layers: Number of transformer blocks.
        n_kv_heads: Number of key-value heads per block.
        d_head: Width of one head.
        seq_len: Number of tokens held.
        bytes_per_value: 2 for 16-bit numbers.

    Returns:
        The byte count, as an int.
    """
    return int(2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per_value)


def count_params(cfg: dict) -> int:
    """Count the trainable parameters of a decoder-only transformer.

    Keys of cfg: d_model, n_layers, n_heads, n_kv_heads, d_ff (the width of each
    feed-forward matrix), vocab, max_positions (0 when positions are rotary),
    gated (three feed-forward matrices instead of two), bias (every linear layer has
    a bias), norm ("layer" has a gain and a bias, "rms" has a gain), and
    tie_embeddings (the output projection reuses the embedding table).

    Every block has two norms, and one final norm follows the last block.

    Raises:
        ValueError: when n_heads does not divide d_model, or n_kv_heads does not
        divide n_heads.
    """
    d, n_heads, n_kv = cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"]
    if d % n_heads != 0 or n_heads % n_kv != 0:
        raise ValueError(f"n_heads={n_heads} must divide d_model={d}, and n_kv_heads={n_kv} must divide n_heads")
    d_head = d // n_heads
    d_ff = cfg["d_ff"]
    attn = d * n_heads * d_head + 2 * d * n_kv * d_head + n_heads * d_head * d
    ffn = (3 if cfg["gated"] else 2) * d * d_ff
    if cfg["bias"]:
        attn += n_heads * d_head + 2 * n_kv * d_head + d
        ffn += (2 * d_ff + d) if cfg["gated"] else (d_ff + d)
    norm = d * (2 if cfg["norm"] == "layer" else 1)
    block = attn + ffn + 2 * norm
    total = cfg["n_layers"] * block + norm
    total += cfg["vocab"] * d + cfg["max_positions"] * d
    if not cfg["tie_embeddings"]:
        total += cfg["vocab"] * d
    return int(total)


# Provided: a ReLU feed-forward sublayer with its backward pass.
def mlp_forward(h: np.ndarray, W1: np.ndarray, W2: np.ndarray):
    a = h @ W1
    return np.maximum(a, 0.0) @ W2, a


def mlp_backward(dout: np.ndarray, W1: np.ndarray, W2: np.ndarray, a: np.ndarray) -> np.ndarray:
    return ((dout @ W2.T) * (a > 0.0)) @ W1.T


def residual_stack(x: np.ndarray, layers: list, kind: str, dout: np.ndarray):
    """Run x through a stack of residual sublayers, then backpropagate dout.

    Each layer is a dict with W1, W2, and gain. With f the ReLU feed-forward
    sublayer (mlp_forward):

        "pre":  x_next = x + f(rmsnorm(x, gain))
        "post": x_next = rmsnorm(x + f(x), gain)

    There is no final norm.

    Args:
        x: Input of shape (T, d).
        layers: List of dicts, applied in order.
        kind: "pre" or "post".
        dout: Gradient of the loss with respect to the stack's output, shape (T, d).

    Returns:
        (out, grads). out has shape (T, d). grads[l] is the gradient of the loss
        with respect to the input of layer l, so grads[0] is the gradient
        with respect to x. There is one entry per layer.
    """
    if kind not in ("pre", "post"):
        raise ValueError(f"kind must be 'pre' or 'post', got {kind!r}")
    caches = []
    for layer in layers:
        W1, W2, g = layer["W1"], layer["W2"], layer["gain"]
        if kind == "pre":
            f, a = mlp_forward(rmsnorm(x, g), W1, W2)
            caches.append((x, a))
            x = x + f
        else:
            f, a = mlp_forward(x, W1, W2)
            z = x + f
            caches.append((z, a))
            x = rmsnorm(z, g)
    out = x
    grads = []
    dx = dout
    for layer, (u, a) in zip(reversed(layers), reversed(caches)):
        W1, W2, g = layer["W1"], layer["W2"], layer["gain"]
        if kind == "pre":
            dn = mlp_backward(dx, W1, W2, a)
            dx = dx + rmsnorm_backward(dn, u, g)[0]
        else:
            dz = rmsnorm_backward(dx, u, g)[0]
            dx = dz + mlp_backward(dz, W1, W2, a)
        grads.append(dx)
    return out, grads[::-1]


# Provided: seeded layers, and the gradient norm that reaches each layer's input.
def make_stack(n_layers: int, d: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [
        {
            "W1": rng.normal(size=(d, 4 * d)) * np.sqrt(2.0 / d),
            "W2": rng.normal(size=(4 * d, d)) * np.sqrt(1.0 / (4 * d)),
            "gain": np.ones(d),
        }
        for _ in range(n_layers)
    ]


def depth_profile(kind: str, n_layers: int, d: int, seed: int, T: int = 16):
    """Return (output rms, first-layer gradient norm over output gradient norm)."""
    rng = np.random.default_rng(seed + 1000)
    x = rng.normal(size=(T, d))
    dout = rng.normal(size=(T, d))
    out, grads = residual_stack(x, make_stack(n_layers, d, seed), kind, dout)
    return float(np.sqrt(np.mean(out**2))), float(np.linalg.norm(grads[0]) / np.linalg.norm(dout))


if __name__ == "__main__":
    row = np.array([1.0, 2.0, 3.0, 6.0])
    print("rmsnorm [1, 2, 3, 6]:", np.round(rmsnorm(row, np.ones(4)), 4))
    llama3 = dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336, vocab=128256,
                  max_positions=0, gated=True, bias=False, norm="rms", tie_embeddings=False)
    print("Llama 3 8B parameters:", count_params(llama3))
    print("KV bytes per token:", kv_cache_bytes(32, 8, 128, 1))
    for kind in ("pre", "post"):
        rms, ratio = depth_profile(kind, 32, 64, 0)
        print(f"{kind}-norm, 32 layers: output rms {rms:.2f}, first-layer gradient / output gradient {ratio:.2f}")
