"""Count the digits the normal equations lose.

You write three functions. The first two measure how hard a problem is. The
third solves the normal equations the fast way, so you can see what that costs.
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


if __name__ == "__main__":
    houses = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    print("condition number of the houses:", condition_number(houses))  # about 6.79

    near = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.1]])
    print("condition number, nearly repeated:", condition_number(near))  # about 127.9
    for y in (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.03])):
        print("weights:", fit_normal_equations(near, y))  # (1, 0), then (0.7, 0.3)

    for kappa in (1e3, 1e7):
        print(
            f"kappa {kappa:.0e}: about {digits_kept(kappa, False):.0f} digits with X, "
            f"{digits_kept(kappa, True):.0f} through X^T X"
        )
