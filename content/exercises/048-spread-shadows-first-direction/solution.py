"""Reference solution: measure spread along a line.

A line through the middle of a cloud turns each point into one score, the
length of its shadow on the line. The variance of the scores says how spread
out the shadows are, and each point's squared length splits into a kept part
(shadow squared) and a lost part (gap squared). You build the pieces that
measure both.
"""

from __future__ import annotations

import numpy as np


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the column mean from X.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        A tuple `(Xc, mean)`. `Xc` is `X` with its column mean subtracted,
        shape (n, d). `mean` is the column mean removed, shape (d,).
    """
    mean = X.mean(axis=0)
    return X - mean, mean


def variance_along(Xc: np.ndarray, u: np.ndarray) -> float:
    """Variance of the scores of centred data along a unit vector.

    Args:
        Xc: Centred data, shape (n, d).
        u: A unit vector, shape (d,).

    Returns:
        The sum of squared scores `Xc @ u`, divided by `n - 1`, as a float.
    """
    scores = Xc @ u
    return float(scores @ scores / (Xc.shape[0] - 1))


def shadow_split(Xc: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Split the total squared length of centred data into kept and lost parts.

    Args:
        Xc: Centred data, shape (n, d).
        u: A unit vector, shape (d,).

    Returns:
        A tuple `(kept, lost)`. `kept` is the sum over points of the squared
        shadow on `u`. `lost` is the sum over points of the squared length of
        what is left after you subtract each shadow, `x - (x @ u) * u`.
    """
    scores = Xc @ u
    kept = float(np.sum(scores**2))
    leftover = Xc - np.outer(scores, u)
    lost = float(np.sum(leftover**2))
    return kept, lost


# Given: it only calls your variance_along once per angle.
def sweep_directions(Xc: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Variance along the unit vector (cos a, sin a), for each angle a in radians.

    Args:
        Xc: Centred data with two columns, shape (n, 2).
        angles: Angles in radians, shape (m,).

    Returns:
        The variances, shape (m,).
    """
    return np.array([variance_along(Xc, np.array([np.cos(a), np.sin(a)])) for a in angles])


if __name__ == "__main__":
    X = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    Xc, mean = center(X)
    print("middle of the cloud:", mean)

    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    w = np.array([1.0, -1.0]) / np.sqrt(2.0)
    print(f"variance along u: {variance_along(Xc, u):.4f}, along w: {variance_along(Xc, w):.4f}")

    kept, lost = shadow_split(Xc, u)
    print(f"line u keeps {kept:.4f} and loses {lost:.4f}")

    angles = np.radians(np.arange(0, 180, 5))
    best = angles[np.argmax(sweep_directions(Xc, angles))]
    print(f"best angle: {np.degrees(best):.0f} degrees")
