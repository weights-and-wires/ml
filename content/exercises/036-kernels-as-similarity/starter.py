"""Kernels as similarity scores.

Fill in the three functions below. `rbf_kernel` and `polynomial_kernel` build
kernel matrices between two sets of points. `poly_features_degree2` builds the
explicit six-column feature map whose dot product the degree-2 polynomial
kernel computes without ever forming those columns.
"""

import numpy as np


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """Return the RBF (Gaussian) kernel matrix between two point sets.

    k(x, x') = exp(-gamma * ||x - x'||^2). A larger gamma gives a narrower bump.

    Args:
        X1: Points, shape (n1, d).
        X2: Points, shape (n2, d).
        gamma: Inverse width of the bump, gamma > 0.

    Returns:
        The kernel matrix, shape (n1, n2). Entries lie in (0, 1].
    """
    raise NotImplementedError("rbf_kernel")


def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, coef0: float = 1.0) -> np.ndarray:
    """Return the polynomial kernel matrix between two point sets.

    k(x, x') = (x . x' + coef0) ** degree, with coef0 >= 0 and degree a positive integer.

    Args:
        X1: Points, shape (n1, d).
        X2: Points, shape (n2, d).
        degree: A positive integer degree.
        coef0: The constant added inside the power.

    Returns:
        The kernel matrix, shape (n1, n2).
    """
    raise NotImplementedError("polynomial_kernel")


def poly_features_degree2(X: np.ndarray) -> np.ndarray:
    """Return the explicit features behind the degree-2 polynomial kernel in 2-D.

    Each point (x1, x2) maps to
    (1, sqrt(2) x1, sqrt(2) x2, x1^2, sqrt(2) x1 x2, x2^2), so that
    phi(x) . phi(x') = (x . x' + 1) ** 2.

    Args:
        X: Points, shape (n, 2).

    Returns:
        The feature matrix, shape (n, 6), columns in the order above.
    """
    raise NotImplementedError("poly_features_degree2")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.array([[1.0, 2.0]])
    x_prime = np.array([[3.0, 1.0]])
    print("polynomial kernel, degree 2, coef0 1:", polynomial_kernel(x, x_prime, 2, 1.0)[0, 0])
    phi_x, phi_xp = poly_features_degree2(x), poly_features_degree2(x_prime)
    print("explicit six-feature dot product:   ", (phi_x @ phi_xp.T)[0, 0])

    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    print("RBF kernel for (0,0) and (1,0), gamma 0.5:", rbf_kernel(a, b, 0.5)[0, 0])

    X = rng.normal(size=(5, 2))
    for gamma in (0.1, 1.0, 10.0):
        K = rbf_kernel(X, X, gamma)
        print(f"gamma={gamma:>4}: mean off-diagonal similarity = {(K.sum() - np.trace(K)) / 20:.3f}")
