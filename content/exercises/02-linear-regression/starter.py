"""Least squares three ways.

Fill in the five functions. Each solves, or measures, the same problem:

    minimise over w:  0.5 * ||X w - y||^2

The point of the exercise is that the three solvers agree in exact arithmetic
and disagree in floating point, by an amount the condition number predicts.

Build the solvers from factorisations. Calling `numpy.linalg.lstsq` or
`numpy.linalg.pinv` skips the exercise, and the checks reject it.
"""

import numpy as np
from scipy.linalg import solve_triangular


def fit_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares through the Gram matrix and a Cholesky factorisation.

    This is the route the lesson tells you not to use. Implement it anyway, so
    the final check can measure what it costs.

    Args:
        X: Design matrix, shape (n, d), assumed to have full column rank.
        y: Targets, shape (n,).

    Returns:
        The coefficient vector, shape (d,).
    """
    raise NotImplementedError("fit_normal_equations")


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve least squares through the thin QR factorisation of X.

    Args:
        X: Design matrix, shape (n, d), assumed to have full column rank.
        y: Targets, shape (n,).

    Returns:
        The coefficient vector, shape (d,).
    """
    raise NotImplementedError("fit_qr")


def fit_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Solve least squares through the SVD, returning the minimum-norm solution.

    Treat a singular value as zero when it is at or below `rcond * s.max()`. When
    X is rank deficient the returned vector must be the element of the solution
    set with the smallest Euclidean norm.

    Args:
        X: Design matrix, shape (n, d). Any rank.
        y: Targets, shape (n,).
        rcond: Relative tolerance below which a singular value counts as zero.

    Returns:
        The coefficient vector, shape (d,).
    """
    raise NotImplementedError("fit_svd")


def residual_orthogonality(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Return the largest absolute entry of X^T (y - X w).

    At the least squares solution this is zero up to rounding, because the
    residual is orthogonal to the column space of X.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: A coefficient vector, shape (d,).

    Returns:
        The infinity norm of the gradient of the loss, as a float.
    """
    raise NotImplementedError("residual_orthogonality")


def condition_number(X: np.ndarray) -> float:
    """Return the spectral condition number of X, sigma_max / sigma_min.

    Args:
        X: Design matrix, shape (n, d).

    Returns:
        The ratio of the largest to the smallest singular value. Return
        `float("inf")` when the smallest singular value is exactly zero.
        A rank-deficient matrix whose smallest singular value is rounding noise
        returns an enormous finite number, which is the correct answer too.
    """
    raise NotImplementedError("condition_number")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0, -0.25])
    y = X @ w_true + 0.1 * rng.normal(size=n)

    print(f"condition number: {condition_number(X):.3f}")
    for name, w in [
        ("normal equations", fit_normal_equations(X, y)),
        ("qr", fit_qr(X, y)),
        ("svd", fit_svd(X, y)),
    ]:
        error = np.abs(w - w_true).max()
        print(
            f"{name:>17}: max coefficient error {error:.2e}, "
            f"orthogonality {residual_orthogonality(X, y, w):.2e}"
        )
