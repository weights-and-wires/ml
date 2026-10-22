"""Rotary position embeddings: turn each query and key by its position."""

import numpy as np


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
