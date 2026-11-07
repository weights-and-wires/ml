"""Kernels and validity: reference solution."""

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
    sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
    # Rounding can leave a tiny negative squared distance for near-identical points.
    return np.exp(-gamma * np.maximum(sq, 0.0))


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
    return (X1 @ X2.T + coef0) ** degree


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
    x1, x2 = X[:, 0], X[:, 1]
    root2 = np.sqrt(2.0)
    return np.stack([np.ones(len(X)), root2 * x1, root2 * x2, x1**2, root2 * x1 * x2, x2**2], axis=1)


def min_eigenvalue(K: np.ndarray) -> float:
    """Return the smallest eigenvalue of a symmetric matrix.

    Args:
        K: A symmetric matrix, shape (n, n).

    Returns:
        The smallest eigenvalue as a Python float.
    """
    return float(np.linalg.eigvalsh(K)[0])


def combination_squared_length(K: np.ndarray, v: np.ndarray) -> float:
    """Return v^T K v, the squared length of the combination of points with weights v.

    If K = Phi Phi^T for real features Phi, this equals ||Phi^T v||^2 and is never negative.

    Args:
        K: A symmetric matrix, shape (n, n).
        v: Weights, shape (n,).

    Returns:
        The scalar v^T K v as a Python float.
    """
    return float(v @ K @ v)


def is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Return whether K can be a Gram matrix: square, symmetric, and PSD.

    Args:
        K: A candidate Gram matrix, shape (n, n).
        tol: Numerical tolerance. K counts as symmetric if every entry differs
            from its transpose by at most `tol` (absolute), and as positive
            semidefinite if its smallest eigenvalue is at least `-tol`.

    Returns:
        True if K passes all three tests, as a Python bool.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        return False
    if not np.allclose(K, K.T, atol=tol, rtol=0.0):
        return False
    return bool(min_eigenvalue(K) >= -tol)


if __name__ == "__main__":
    x = np.array([[1.0, 2.0]])
    x_prime = np.array([[3.0, 1.0]])
    print("polynomial kernel, degree 2, coef0 1:", polynomial_kernel(x, x_prime, 2, 1.0)[0, 0])
    phi_x, phi_xp = poly_features_degree2(x), poly_features_degree2(x_prime)
    print("explicit six-feature dot product:   ", (phi_x @ phi_xp.T)[0, 0])
    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    print("RBF kernel for (0,0) and (1,0), gamma 0.5:", rbf_kernel(a, b, 0.5)[0, 0])

    invented = np.array([[1.0, 2.0], [2.0, 1.0]])
    print("smallest eigenvalue of the invented table:", min_eigenvalue(invented))
    print("squared length of the combination (1, -1):", combination_squared_length(invented, np.array([1.0, -1.0])))
    print("invented table is PSD:", is_psd(invented))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    for c in (1.0, 0.0, -2.0):
        K = polynomial_kernel(X, X, 2, c)
        print(f"degree-2 polynomial, coef0={c:>4}: smallest eigenvalue {min_eigenvalue(K):.3g}, PSD: {is_psd(K)}")
    print("RBF Gram matrix is PSD:", is_psd(rbf_kernel(X, X, 1.0)))
