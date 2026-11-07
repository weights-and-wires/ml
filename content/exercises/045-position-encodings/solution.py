"""Position encodings: sinusoidal clock hands, and rotary embeddings."""

import numpy as np


def _speeds(d: int) -> np.ndarray:
    return 10000.0 ** (-2.0 * np.arange(d // 2) / d)


def sinusoidal_encoding(T: int, d: int) -> np.ndarray:
    angles = np.arange(T, dtype=np.float64)[:, None] * _speeds(d)[None, :]  # (T, d/2)
    pe = np.empty((T, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def offset_rotation(k: int, d: int) -> np.ndarray:
    R = np.zeros((d, d))
    for i, omega in enumerate(_speeds(d)):
        c, s = np.cos(k * omega), np.sin(k * omega)
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, s], [-s, c]]
    return R


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
    pe = sinusoidal_encoding(T=3, d=4)
    print("Encoding at width 4 (rows are positions 0, 1, 2):")
    print(np.round(pe, 4))

    R = offset_rotation(k=1, d=4)
    print("Moving position 1 to position 2 with offset_rotation(1, 4):")
    print(np.round(R @ pe[1], 4), "versus", np.round(pe[2], 4))

    q = k = np.array([1.0, 0.0])
    print("Rotary, q = k = [1, 0], positions 0 and 1:", np.round(rope_score(q, k, 0, 1), 4))
    print("Rotary, positions 1 and 2 (the same distance):", np.round(rope_score(q, k, 1, 2), 4))

    rng = np.random.default_rng(0)
    q, k = rng.normal(size=8), rng.normal(size=8)
    scores = [rope_score(q, k, m, m + 3) for m in (0, 5, 40)]
    print("Random width-8 vectors at offset 3, from three starting positions:", np.round(scores, 4))
