"""Reference solution for the merging exercise."""

import numpy as np


# Written in the chapter "A low-rank update: build the LoRA layer".
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


def extra_multiplies(d_in: int, d_out: int, r: int) -> int:
    return int(r * d_in + d_out * r)


def merged_matches_split(
    x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> float:
    merged = merge_lora(W, A, B, alpha, r)
    split = lora_forward(x, W, A, B, alpha, r)
    return float(np.abs(x @ merged.T - split).max())


if __name__ == "__main__":
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    print("merged weight (expect [[3, 4], [0, 1]]):")
    print(merge_lora(W, A, B, 2.0, 1))

    rng = np.random.default_rng(0)
    d_in, d_out, r = 64, 48, 4
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in)) / np.sqrt(d_in)
    B = rng.normal(size=(d_out, r))
    x = rng.normal(size=(5, d_in))
    print("largest gap, merged vs split:", merged_matches_split(x, W, A, B, 8.0, r))

    extra = extra_multiplies(4096, 4096, 8)
    print(f"extra multiplies at 4096 x 4096, r = 8: {extra:,} ({extra / 4096**2:.2%} of the base)")
