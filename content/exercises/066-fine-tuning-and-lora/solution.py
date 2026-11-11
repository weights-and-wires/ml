"""Reference solution for the fine-tuning and LoRA exercise."""

import numpy as np


def adam_bytes_breakdown() -> dict:
    return {
        "weight": 2,
        "gradient": 2,
        "master_weight": 4,
        "first_average": 4,
        "second_average": 4,
    }


def trainable_fraction_memory(total_params: int, trained_params: int) -> int:
    if trained_params > total_params:
        raise ValueError("trained_params cannot exceed total_params")
    return 16 * trained_params + 2 * (total_params - trained_params)


def lora_parameter_count(d_in: int, d_out: int, r: int) -> int:
    return int(r * (d_in + d_out))


def lora_init(d_in: int, d_out: int, r: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(r, d_in)) / np.sqrt(d_in)
    B = np.zeros((d_out, r))
    return A, B


def lora_forward(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    base = x @ W.T
    update = (x @ A.T) @ B.T
    return base + (alpha / r) * update


def merge_lora(
    W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    return W + (alpha / r) * (B @ A)


def best_rank_approximation(delta_w: np.ndarray, r: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(delta_w, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def energy_captured(delta_w: np.ndarray, r: int) -> float:
    s = np.linalg.svd(delta_w, compute_uv=False)
    return float((s[:r] ** 2).sum() / (s**2).sum())


def lora_gradients(
    W: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
    r: int,
    target: np.ndarray,
) -> tuple:
    G = merge_lora(W, A, B, alpha, r) - target
    scale = alpha / r
    dA = scale * (B.T @ G)
    dB = scale * (G @ A.T)
    return dA, dB


if __name__ == "__main__":
    pieces = adam_bytes_breakdown()
    print("bytes per trained parameter:", pieces, "->", sum(pieces.values()))
    n = 7_000_000_000
    frozen = trainable_fraction_memory(n, n // 100)
    print(f"train 1% of {n:,} and freeze the rest: {frozen / 1e9:.2f} GB")
    print("4096 x 4096 at rank 8 trains:", lora_parameter_count(4096, 4096, 8), "numbers")

    # The 2 by 2 example from the chapter.
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    print("hand example (expect [7, 1]):", lora_forward(x, W, A, B, alpha=2.0, r=1))
    print("merged weight (expect [[3, 4], [0, 1]]):")
    print(merge_lora(W, A, B, 2.0, 1))

    # A freshly initialised layer matches the frozen layer.
    rng = np.random.default_rng(0)
    d_in, d_out, r = 64, 48, 4
    W2 = rng.normal(size=(d_out, d_in))
    A2, B2 = lora_init(d_in, d_out, r, seed=1)
    x2 = rng.normal(size=(5, d_in))
    gap = np.abs(lora_forward(x2, W2, A2, B2, 8.0, r) - x2 @ W2.T).max()
    print("step-zero difference from the frozen layer:", gap)

    # Truncation and training.
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    approx = best_rank_approximation(M, 1)
    print("best rank-1 approximation of [[2, 1], [1, 2]]:")
    print(approx)
    print("error:", np.linalg.norm(M - approx), " energy kept:", energy_captured(M, 1))

    target = np.array([[1.0, 1.0], [2.0, 0.0]])
    dA, dB = lora_gradients(W, A, B, 2.0, 1, target)
    print("dA (expect [[4, 6]]):", dA)
    print("dB (expect [[16], [0]]):", dB.ravel())
