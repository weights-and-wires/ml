"""Modern transformer choices: RMSNorm, grouped-query attention, parameter counts, and norm placement.

Arrays are unbatched unless a docstring says otherwise. Write the six functions that
raise NotImplementedError. The helpers marked "Provided" are already done.
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
    raise NotImplementedError("rmsnorm")


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
    raise NotImplementedError("rmsnorm_backward")


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
    raise NotImplementedError("gqa_attention")


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
    raise NotImplementedError("kv_cache_bytes")


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
    raise NotImplementedError("count_params")


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
    raise NotImplementedError("residual_stack")


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
