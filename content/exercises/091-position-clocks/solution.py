"""Sinusoidal position encodings: one clock hand per pair of columns."""

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


def add_positions(x: np.ndarray, pe: np.ndarray) -> np.ndarray:
    return x + pe[: x.shape[0]]


if __name__ == "__main__":
    pe = sinusoidal_encoding(T=3, d=4)
    print("Encoding at width 4 (rows are positions 0, 1, 2):")
    print(np.round(pe, 4))

    R = offset_rotation(k=1, d=4)
    print("Moving position 1 to position 2 with offset_rotation(1, 4):")
    print(np.round(R @ pe[1], 4), "versus", np.round(pe[2], 4))

    tokens = np.array([[1.0, 0.0, 0.0, 0.0]] * 3)  # the same token three times
    print("Same token at three positions, after add_positions:")
    print(np.round(add_positions(tokens, pe), 4))
