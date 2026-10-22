"""Stable least squares with QR and the SVD.

You write three functions. Neither solver may form X^T X, and the checks reject
source that calls numpy's `lstsq` or `pinv`.
"""

import numpy as np
from scipy.linalg import solve_triangular


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares through the reduced QR factorisation of X.

    Args:
        X: Table of inputs, shape (n, d), assumed to have full column rank.
        y: Targets, shape (n,).

    Returns:
        The weights, shape (d,). They solve R w = Q^T y.
    """
    raise NotImplementedError("fit_qr")


def fit_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Solve least squares through the SVD, returning the minimum-norm weights.

    Treat a singular value as zero when it is at or below rcond * s[0]. When
    columns repeat, many weights fit equally well, and the returned vector must
    be the shortest of them.

    Args:
        X: Table of inputs, shape (n, d). Any rank.
        y: Targets, shape (n,).
        rcond: Relative cut-off below which a singular value counts as zero.

    Returns:
        The weights, shape (d,).
    """
    raise NotImplementedError("fit_svd")


def rank(X: np.ndarray, rcond: float = 1e-12) -> int:
    """Count the singular values of X above rcond * s[0].

    Args:
        X: Table of inputs, shape (n, d).
        rcond: Relative cut-off.

    Returns:
        The number of directions the columns really use, as an int. An all-zero
        matrix returns 0.
    """
    raise NotImplementedError("rank")


if __name__ == "__main__":
    houses = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 2.0])
    print("QR:", fit_qr(houses, y))  # about [0.6667 0.5]
    print("SVD:", fit_svd(houses, y))  # about [0.6667 0.5]

    repeated = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    ones = np.ones(3)
    print("rank of the repeated columns:", rank(repeated))  # 1
    print("shortest weights:", fit_svd(repeated, ones))  # about [0.0857 0.1714]
