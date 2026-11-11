"""Position encodings: sinusoidal clock hands, and rotary embeddings."""

import numpy as np


def sinusoidal_encoding(T: int, d: int) -> np.ndarray:
    """Return the sinusoidal position encoding, one row per position.

    Hand i, for i = 0, ..., d/2 - 1, turns at speed omega_i = 10000 ** (-2i / d).
    At position p, column 2i holds sin(p * omega_i) and column 2i + 1 holds
    cos(p * omega_i). d must be even.

    Args:
        T: Number of positions.
        d: Encoding width, even.

    Returns:
        Array of shape (T, d). Row 0 is [0, 1, 0, 1, ...].
    """
    raise NotImplementedError("sinusoidal_encoding")


def offset_rotation(k: int, d: int) -> np.ndarray:
    """Return the matrix R_k with R_k @ PE(p) == PE(p + k) for every p.

    The matrix is block diagonal. The block at rows and columns 2i and 2i + 1
    turns hand i by the angle k * omega_i.

    Args:
        k: The offset, any integer.
        d: Encoding width, even.

    Returns:
        Array of shape (d, d).
    """
    raise NotImplementedError("offset_rotation")


def rope_tables(T: int, d: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the cosine and sine tables for positions 0, ..., T - 1.

    Pair j of a width-d vector turns at speed theta ** (-2j / d), for
    j = 0, ..., d/2 - 1. At position p it turns by the angle p * speed_j.
    Each speed is stored twice, once for entry j and once for entry j + d/2,
    so the tables line up with both halves of a vector. d must be even.

    Args:
        T: Number of positions.
        d: Head width, even.
        theta: The base of the speeds.

    Returns:
        A pair (cos, sin). Each has shape (T, d). Row 0 of cos is all ones and
        row 0 of sin is all zeros.
    """
    raise NotImplementedError("rope_tables")


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Turn x by the angles stored in cos and sin, with the rotate-half layout.

    Split the last axis of x into halves x1 and x2. Entry j of x1 and entry j
    of x2 form one pair. The result is

        x * cos + rotate_half(x) * sin,  rotate_half(x) = [-x2, x1]

    cos and sin broadcast against x. x can be one vector of shape (d,), a
    sequence of shape (T, d), or a stack of heads of shape (heads, T, d).

    Args:
        x: Array whose last axis has even size d.
        cos: Array of cosines, broadcastable against x, last axis size d.
        sin: Array of sines, broadcastable against x, last axis size d.

    Returns:
        Array with the same shape as x.
    """
    raise NotImplementedError("apply_rope")


def rope_score(q: np.ndarray, k: np.ndarray, m: int, n: int, theta: float = 10000.0) -> float:
    """Return the dot product of q turned to position m and k turned to position n.

    Args:
        q: Query vector of shape (d,).
        k: Key vector of shape (d,).
        m: Position of the query, zero or more.
        n: Position of the key, zero or more.
        theta: The base of the speeds.

    Returns:
        The score as a Python float.
    """
    raise NotImplementedError("rope_score")


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
