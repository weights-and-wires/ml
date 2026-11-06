"""Valid kernels and the positive semidefinite test: reference solution.

Fill in the three functions below. `min_eigenvalue` reports the smallest
eigenvalue of a symmetric matrix. `combination_squared_length` computes
v^T K v, the squared length of the combination of points with weights v.
`is_psd` uses both ideas to decide whether a matrix can be a Gram matrix.
"""

import numpy as np


# Implemented in the exercise for "Kernels: a similarity score that hides a bigger
# feature space". Provided here so you can build Gram matrices to test.
def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


# Implemented in the exercise for "Kernels: a similarity score that hides a bigger
# feature space".
def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, coef0: float = 1.0) -> np.ndarray:
    return (X1 @ X2.T + coef0) ** degree


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
    invented = np.array([[1.0, 2.0], [2.0, 1.0]])
    print("smallest eigenvalue of the invented table:", min_eigenvalue(invented))
    witness = np.array([1.0, -1.0])
    print("squared length of the combination (1, -1):", combination_squared_length(invented, witness))
    print("invented table is PSD:", is_psd(invented))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    for c in (1.0, 0.0, -2.0):
        K = polynomial_kernel(X, X, 2, c)
        print(f"degree-2 polynomial, coef0={c:>4}: smallest eigenvalue {min_eigenvalue(K):.3g}, PSD: {is_psd(K)}")
    print("RBF Gram matrix is PSD:", is_psd(rbf_kernel(X, X, 1.0)))
