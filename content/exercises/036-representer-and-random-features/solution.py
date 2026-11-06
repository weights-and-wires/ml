"""The representer theorem and random Fourier features: reference solution.

Fill in the three functions below. `representer_check` evaluates the
regularised objective of a candidate function given by weights over a list of
centres, so you can test that extra centres never help.
`random_fourier_features` builds explicit features whose dot products
approximate the RBF kernel. `ridge_on_features` fits ridge regression on them.
"""

import numpy as np


# Implemented in the exercise for "Kernels: a similarity score that hides a bigger
# feature space". Provided here so you can build kernel matrices.
def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


# Implemented in the exercise for "Kernel ridge regression: a weighted sum of
# similarities".
def kernel_ridge_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = K.shape[0]
    L = np.linalg.cholesky(K + lam * np.eye(n))
    z = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, z)


def representer_check(K: np.ndarray, y: np.ndarray, lam: float, coeffs: np.ndarray) -> float:
    """Return the regularised objective of a candidate function.

    The candidate is sum_j coeffs[j] * k(x, z_j) over N centres, where the
    first len(y) centres are the training points and the rest are extra
    centres. The objective is

        sum_{i < n} (y_i - (K @ coeffs)_i)^2 + lam * coeffs @ K @ coeffs.

    Args:
        K: Gram matrix of all N centres, shape (N, N).
        y: Training targets, shape (n,), with n <= N.
        lam: Ridge strength.
        coeffs: Weights on the N centres, shape (N,).

    Returns:
        The objective as a float. The error is a sum, not a mean.
    """
    n = len(y)
    predictions = (K @ coeffs)[:n]
    loss = float(np.sum((y - predictions) ** 2))
    return loss + lam * float(coeffs @ K @ coeffs)


def random_fourier_features(X: np.ndarray, D: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Return random Fourier features whose dot products approximate the RBF kernel.

    Z = sqrt(2 / D) * cos(X @ omega.T + phase), where omega has shape (D, d)
    with independent normal entries of variance 2 * gamma, and phase has shape
    (D,) with entries uniform on [0, 2 pi). All randomness comes from `rng`, and
    the shapes of omega and phase depend only on D and d, so a generator with
    the same seed gives the same feature map for any number of rows.

    Args:
        X: Points, shape (n, d).
        D: Number of random features.
        gamma: The RBF parameter of the kernel being approximated.
        rng: A numpy random generator.

    Returns:
        The feature matrix Z, shape (n, D), with Z @ Z.T approximately equal to
        the RBF Gram matrix of X.
    """
    d = X.shape[1]
    omega = rng.normal(size=(D, d)) * np.sqrt(2.0 * gamma)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ omega.T + phase)


def ridge_on_features(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Return the ridge regression weights on a feature matrix.

    Solves (Z^T Z + lam I) w = Z^T y.

    Args:
        Z: Features, shape (n, D).
        y: Targets, shape (n,).
        lam: Ridge strength, lam > 0.

    Returns:
        The weights w, shape (D,).
    """
    D = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + lam * np.eye(D), Z.T @ y)


if __name__ == "__main__":
    # The worked example: K = I_2, one training target y = 2, lam = 1.
    K = np.eye(2)
    y = np.array([2.0])
    print("J for c = (1, 0):", representer_check(K, y, 1.0, np.array([1.0, 0.0])), "(expected 2)")
    print("J for c = (1, 1):", representer_check(K, y, 1.0, np.array([1.0, 1.0])), "(expected 3)")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    gamma = 0.5
    K = rbf_kernel(X, X, gamma)
    for D in (10, 160, 2560):
        Z = random_fourier_features(X, D, gamma, np.random.default_rng(1))
        print(f"D={D:>5}: mean absolute error against the exact RBF Gram matrix = {np.abs(K - Z @ Z.T).mean():.4f}")

    x = np.sort(rng.uniform(0, 6, size=40))[:, None]
    y = np.sin(x[:, 0]) + 0.1 * rng.normal(size=40)
    x_test = np.linspace(0, 6, 25)[:, None]
    alpha = kernel_ridge_fit(rbf_kernel(x, x, 1.0), y, 0.1)
    exact = rbf_kernel(x_test, x, 1.0) @ alpha
    Z = random_fourier_features(x, 2000, 1.0, np.random.default_rng(7))
    Z_test = random_fourier_features(x_test, 2000, 1.0, np.random.default_rng(7))
    w = ridge_on_features(Z, y, 0.1)
    print(f"largest gap between random-feature ridge and exact kernel ridge: {np.abs(Z_test @ w - exact).max():.4f}")
