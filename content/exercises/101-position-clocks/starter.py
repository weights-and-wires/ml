"""Sinusoidal position encodings: one clock hand per pair of columns."""

import numpy as np


def sinusoidal_encoding(T: int, d: int) -> np.ndarray:
    """Return the sinusoidal position encoding for positions 0, ..., T - 1.

    Hand i (for i = 0, ..., d/2 - 1) turns at speed omega_i = 10000 ** (-2i / d).
    Row p, column 2i holds sin(p * omega_i), and column 2i + 1 holds
    cos(p * omega_i). d must be even.

    Args:
        T: Number of positions.
        d: Width of the encoding, even.

    Returns:
        Array of shape (T, d).
    """
    raise NotImplementedError("sinusoidal_encoding")


def offset_rotation(k: int, d: int) -> np.ndarray:
    """Return the (d, d) matrix R with R @ PE(p) == PE(p + k) for every p.

    R is block diagonal. The 2 by 2 block at rows and columns 2i and 2i + 1
    turns hand i by the angle k * omega_i:

        [[ cos(k * omega_i),  sin(k * omega_i)],
         [-sin(k * omega_i),  cos(k * omega_i)]]

    Args:
        k: The offset in positions. It can be zero or negative.
        d: Width of the encoding, even.

    Returns:
        Array of shape (d, d).
    """
    raise NotImplementedError("offset_rotation")


def add_positions(x: np.ndarray, pe: np.ndarray) -> np.ndarray:
    """Add the position encoding to a sequence of token vectors.

    Args:
        x: Token vectors of shape (T, d).
        pe: Position encoding of shape (T_max, d) with T_max >= T.

    Returns:
        A new array of shape (T, d) equal to x + pe[:T]. The input x must not
        change.
    """
    raise NotImplementedError("add_positions")


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
