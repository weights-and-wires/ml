"""Rotary position embeddings: turn each query and key by its position."""

import numpy as np


def rope_tables(T: int, d: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    speeds = theta ** (-2.0 * np.arange(d // 2) / d)             # (d/2,)
    angles = np.arange(T, dtype=np.float64)[:, None] * speeds[None, :]  # (T, d/2)
    cos = np.concatenate([np.cos(angles)] * 2, axis=-1)          # (T, d)
    sin = np.concatenate([np.sin(angles)] * 2, axis=-1)
    return cos, sin


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return x * cos + _rotate_half(x) * sin


def rope_score(q: np.ndarray, k: np.ndarray, m: int, n: int, theta: float = 10000.0) -> float:
    cos, sin = rope_tables(max(m, n) + 1, q.shape[-1], theta)
    q_turned = apply_rope(q, cos[m], sin[m])
    k_turned = apply_rope(k, cos[n], sin[n])
    return float(q_turned @ k_turned)


if __name__ == "__main__":
    cos, sin = rope_tables(T=3, d=2)
    print("Width 2, positions 0, 1, 2 (cos then sin):")
    print(np.round(cos[:, 0], 4), np.round(sin[:, 0], 4))

    q = k = np.array([1.0, 0.0])
    print("q = k = [1, 0], turned to positions 0 and 1:", np.round(rope_score(q, k, 0, 1, 10000.0), 4))
    print("Turned to positions 1 and 2 (the same distance):", np.round(rope_score(q, k, 1, 2, 10000.0), 4))

    rng = np.random.default_rng(0)
    q, k = rng.normal(size=8), rng.normal(size=8)
    scores = [rope_score(q, k, m, m + 3) for m in (0, 5, 40)]
    print("Random width-8 vectors at offset 3, from three starting positions:", np.round(scores, 4))
