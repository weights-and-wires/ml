"""A LoRA layer, its accounting, and the low-rank approximation it leans on.

Fill in the five functions below. `W` is always the frozen base weight, shape
(d_out, d_in), matching the `nn.Linear` convention y = x W^T. `A` is shape
(r, d_in) and `B` is shape (d_out, r), so that B @ A has the same shape as W.
"""

import numpy as np


def lora_forward(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    """Compute the LoRA-adapted forward pass x W^T + (alpha / r) x A^T B^T.

    Args:
        x: Input batch, shape (batch, d_in).
        W: Frozen base weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The LoRA scaling numerator.
        r: The rank. Must equal A.shape[0] and B.shape[1].

    Returns:
        The layer's output, shape (batch, d_out).
    """
    raise NotImplementedError("lora_forward")


def lora_parameter_count(d_in: int, d_out: int, r: int) -> int:
    """Return the number of trainable parameters in A and B combined.

    Args:
        d_in: Input dimension.
        d_out: Output dimension.
        r: Rank.

    Returns:
        r * (d_in + d_out), as an int.
    """
    raise NotImplementedError("lora_parameter_count")


def merge_lora(
    W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    """Return the merged effective weight W + (alpha / r) * B @ A.

    Args:
        W: Frozen base weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The LoRA scaling numerator.
        r: The rank.

    Returns:
        The merged weight, shape (d_out, d_in), such that x @ merged.T equals
        lora_forward(x, W, A, B, alpha, r) for any x.
    """
    raise NotImplementedError("merge_lora")


def optimiser_memory_bytes(params: int, mode: str) -> int:
    """Return Adam optimizer memory for `params` trainable parameters.

    Counts an fp16 parameter copy, an fp16 gradient, an fp32 master copy, and
    Adam's two fp32 moments: 16 bytes per trainable parameter, for either
    `mode`. `mode` is `"full"` or `"lora"`, documenting which parameter count
    is being charged; it does not change the per-parameter rate, since Adam's
    bookkeeping cost is the same whether the parameters belong to a full
    weight matrix or a LoRA adapter.

    Args:
        params: Number of trainable parameters.
        mode: `"full"` or `"lora"`.

    Returns:
        The optimizer memory in bytes, as an int.
    """
    raise NotImplementedError("optimiser_memory_bytes")


def best_rank_approximation(delta_w: np.ndarray, r: int) -> np.ndarray:
    """Return the closest rank-r matrix to delta_w in Frobenius norm.

    Args:
        delta_w: A matrix, shape (d_out, d_in).
        r: The target rank, r <= min(delta_w.shape).

    Returns:
        The truncated-SVD reconstruction of delta_w at rank r, shape
        (d_out, d_in).
    """
    raise NotImplementedError("best_rank_approximation")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d_in, d_out, r, alpha = 32, 24, 4, 8.0

    W = rng.normal(size=(d_out, d_in)) * 0.1
    A = rng.normal(size=(r, d_in)) * 0.1
    B = np.zeros((d_out, r))
    x = rng.normal(size=(5, d_in))

    print("With B initialised to zero, LoRA matches the base layer:")
    print(f"  max difference: {np.abs(lora_forward(x, W, A, B, alpha, r) - x @ W.T).max():.2e}")

    B = rng.normal(size=(d_out, r)) * 0.1
    y_lora = lora_forward(x, W, A, B, alpha, r)
    W_merged = merge_lora(W, A, B, alpha, r)
    y_merged = x @ W_merged.T
    print(f"\nlora_forward vs. merged weight, max difference: {np.abs(y_lora - y_merged).max():.2e}")

    full_params = d_in * d_out
    lora_params = lora_parameter_count(d_in, d_out, r)
    print(f"\nfull fine-tuning parameters: {full_params:,}")
    print(f"LoRA parameters (r={r}): {lora_params:,}")
    print(f"saving: {1 - lora_params / full_params:.4%}")

    print(f"\noptimizer memory, full: {optimiser_memory_bytes(full_params, 'full'):,} bytes")
    print(f"optimizer memory, lora: {optimiser_memory_bytes(lora_params, 'lora'):,} bytes")

    delta_w = rng.normal(size=(d_out, d_in))
    approx = best_rank_approximation(delta_w, r)
    error = np.linalg.norm(delta_w - approx, ord="fro")
    print(f"\nrank-{r} approximation error: {error:.4f}")
