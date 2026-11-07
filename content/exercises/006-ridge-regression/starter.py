"""Ridge regression: score a fit with a fee on large weights, solve for the bottom, then read every lambda off one SVD.

Fill in the six functions below, in order. The first three score a set of
weights, give the slope of that score, and solve for the weights where the slope
is zero. The last three scale each direction of the data, turn those scales
into the ridge weights for a whole list of lambdas, and count how many
directions the fit still uses.
"""

from __future__ import annotations

import numpy as np


def ridge_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Score the weights: half the squared misses plus half lambda times the squared length.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        lam: Ridge fee, lam >= 0.

    Returns:
        0.5 * ||X w - y||^2 + 0.5 * lam * ||w||^2 as a Python float.
    """
    raise NotImplementedError("ridge_objective")


def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Return the gradient of the ridge score with respect to the weights.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        lam: Ridge fee, lam >= 0.

    Returns:
        The gradient, shape (d,).
    """
    raise NotImplementedError("ridge_gradient")


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve for the weights at the bottom of the ridge bowl with one linear system.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Ridge fee, lam >= 0.

    Returns:
        The weights where the gradient is zero, shape (d,).
    """
    raise NotImplementedError("ridge_closed_form")


def shrink_factors(s: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Return s / (s**2 + lambda) for every singular value and every lambda.

    Args:
        s: Singular values, shape (r,).
        lambdas: Ridge fees, shape (m,), each >= 0.

    Returns:
        An array of shape (m, r) whose entry [j, i] is s[i] / (s[i]**2 + lambdas[j]).
        When both s[i] and lambdas[j] are 0, return 0 for that entry instead of 0 / 0.
    """
    raise NotImplementedError("shrink_factors")


def ridge_svd_path(
    U: np.ndarray, s: np.ndarray, Vt: np.ndarray, y: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    """Compute the ridge weights at every lambda from one thin SVD of X.

    X = U @ diag(s) @ Vt. For each lambda the weights are
    V @ diag(s / (s**2 + lambda)) @ U^T @ y. Compute U^T y once and reuse it.

    Args:
        U: Left singular vectors, shape (n, r).
        s: Singular values, shape (r,).
        Vt: Right singular vectors transposed, shape (r, d).
        y: Targets, shape (n,).
        lambdas: Ridge fees, shape (m,).

    Returns:
        The weights at every lambda, shape (m, d).
    """
    raise NotImplementedError("ridge_svd_path")


def effective_degrees_of_freedom(s: np.ndarray, lam) -> np.ndarray | float:
    """Count the directions the fit still uses: the sum of s**2 / (s**2 + lam).

    Args:
        s: Singular values, shape (r,).
        lam: One ridge fee (a float) or several (an array of shape (m,)).

    Returns:
        A float when lam is a single number, or an array of shape (m,) with one
        count per lambda. A zero singular value at lam = 0 contributes 0, not NaN.
    """
    raise NotImplementedError("effective_degrees_of_freedom")


if __name__ == "__main__":
    # The two-point example: one weight, labels 0 and 1.
    X = np.array([[0.0], [0.1]])
    y = np.array([0.0, 1.0])
    print("lambda   slope   score at the slope")
    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        w = ridge_closed_form(X, y, lam)
        print(f"{lam:6.2f}  {w[0]:6.3f}  {ridge_objective(X, y, w, lam):8.4f}")

    # Two identical columns: least squares cannot choose, ridge splits evenly.
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 2.0])
    print("identical columns, lambda = 1:", ridge_closed_form(X, y, 1.0))

    # A strong direction and a weak one: the whole path from one SVD.
    X = np.array([[3.0, 3.0], [-0.1, 0.1]])
    y = np.array([6.0, 0.4])
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    path = ridge_svd_path(U, s, Vt, y, lambdas)
    dof = effective_degrees_of_freedom(s, lambdas)
    print("singular values:", np.round(s, 4))
    print("lambda    w1       w2     degrees of freedom")
    for lam, w, df in zip(lambdas, path, dof):
        print(f"{lam:6.2f} {w[0]:8.3f} {w[1]:8.3f} {df:10.3f}")
    print("same as one solve per lambda:", np.allclose(path[3], ridge_closed_form(X, y, 1.0)))
