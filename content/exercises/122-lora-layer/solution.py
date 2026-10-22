"""Reference solution for the LoRA layer exercise."""

import numpy as np


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


if __name__ == "__main__":
    # The 2 by 2 example from the chapter.
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    print("hand example (expect [7, 1]):", lora_forward(x, W, A, B, alpha=2.0, r=1))

    # A freshly initialised layer matches the frozen layer.
    rng = np.random.default_rng(0)
    d_in, d_out, r = 64, 48, 4
    W = rng.normal(size=(d_out, d_in))
    A, B = lora_init(d_in, d_out, r, seed=1)
    x = rng.normal(size=(5, d_in))
    gap = np.abs(lora_forward(x, W, A, B, 8.0, r) - x @ W.T).max()
    print("step-zero difference from the frozen layer:", gap)

    full = d_in * d_out
    small = lora_parameter_count(d_in, d_out, r)
    print(f"trained numbers: full {full:,}, LoRA {small:,} ({full / small:.1f} times fewer)")
    print("4096 x 4096 at rank 8:", lora_parameter_count(4096, 4096, 8))
