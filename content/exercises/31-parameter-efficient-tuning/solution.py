"""Reference solution for the parameter-efficient fine-tuning exercise."""

import numpy as np


def lora_forward(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    base = x @ W.T
    update = (x @ A.T) @ B.T
    return base + (alpha / r) * update


def lora_parameter_count(d_in: int, d_out: int, r: int) -> int:
    return r * (d_in + d_out)


def merge_lora(
    W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    return W + (alpha / r) * (B @ A)


def optimiser_memory_bytes(params: int, mode: str) -> int:
    if mode not in ("full", "lora"):
        raise ValueError(f"mode must be 'full' or 'lora', got {mode!r}")
    return 16 * params


def best_rank_approximation(delta_w: np.ndarray, r: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(delta_w, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r, :]


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
