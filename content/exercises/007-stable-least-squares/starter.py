"""Lose digits, then keep them with QR and the SVD.

You write six functions. The first three measure how hard a problem is and
solve it the fast way, through the normal equations, so you can see what that
costs. The last three solve it without forming X^T X. The checks reject source
that calls numpy's `lstsq` or `pinv`.
"""

import numpy as np
from scipy.linalg import solve_triangular


# This helper is written for you. It builds a table with an exact condition
# number by choosing its singular values and two random rotations.
def make_ill_conditioned(n: int, d: int, kappa: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = np.logspace(0.0, -np.log10(kappa), d)
    return u @ np.diag(s) @ v.T


def condition_number(X: np.ndarray) -> float:
    """Return sigma_max / sigma_min for X.

    Args:
        X: Table of inputs, shape (n, d).

    Returns:
        The ratio of the largest to the smallest singular value, as a float.
        Return float("inf") when the smallest singular value is exactly 0.
    """
    raise NotImplementedError("condition_number")


def digits_kept(kappa: float, squared: bool) -> float:
    """Estimate how many of the 16 digits of a double survive.

    A condition number of 10**k loses about k digits. Working with X^T X squares
    the condition number, so it loses about 2k.

    Args:
        kappa: The condition number of X.
        squared: True to estimate for a solve through X^T X.

    Returns:
        max(0, 16 - k), or max(0, 16 - 2k) when squared, where k = log10(kappa).
    """
    raise NotImplementedError("digits_kept")


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve X^T X w = X^T y with a Cholesky factor and two triangular solves.

    Args:
        X: Table of inputs, shape (n, d), assumed to have full column rank.
        y: Targets, shape (n,).

    Returns:
        The weights, shape (d,).
    """
    raise NotImplementedError("fit_normal_equations")


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
    print("condition number of the houses:", condition_number(houses))
    print("normal equations:", fit_normal_equations(houses, y))
    print("QR:", fit_qr(houses, y))
    print("SVD:", fit_svd(houses, y))

    near = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]])
    for t in (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.03])):
        print("weights:", fit_normal_equations(near, t))

    repeated = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    print("rank of the repeated columns:", rank(repeated))
    print("shortest weights:", fit_svd(repeated, np.ones(3)))  # about [0.0857 0.1714]
