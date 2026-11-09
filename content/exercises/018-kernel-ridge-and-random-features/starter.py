"""Kernel ridge regression, the representer theorem, and random Fourier features.

Fill in the six functions below. `rbf_kernel` is already written, from the chapter on
kernels. A kernel model predicts sum_i alpha_i k(x, x_i), and the random features
replace the kernel with explicit columns.
"""

import numpy as np


# Written in the chapter "Kernels: similarity scores and what makes one valid".
def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def kernel_ridge_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Return the kernel ridge weights alpha that solve (K + lam I) alpha = y.

    The fitted model predicts sum_i alpha_i k(x, x_i). Solve with a Cholesky
    factor of K + lam I rather than an explicit inverse.

    Args:
        K: Gram matrix of the training points, shape (n, n), positive semidefinite.
        y: Training targets, shape (n,).
        lam: Ridge strength, lam > 0.

    Returns:
        The weights alpha, shape (n,).
    """
    raise NotImplementedError("kernel_ridge_fit")


def kernel_ridge_predict(K_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return predictions sum_i alpha_i k(x, x_i) for each query row.

    Args:
        K_test: Similarities from each query to each training point,
            shape (n_test, n_train).
        alpha: Weights from `kernel_ridge_fit`, shape (n_train,).

    Returns:
        Predictions, shape (n_test,).
    """
    raise NotImplementedError("kernel_ridge_predict")


def implied_weights(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return the feature-space weights w = X^T alpha for a linear kernel.

    When K = X X^T, kernel ridge regression and ordinary ridge regression give
    the same predictions, and this w is the ordinary ridge weight vector.

    Args:
        X: Training inputs, shape (n, d).
        alpha: Weights from `kernel_ridge_fit`, shape (n,).

    Returns:
        The weight vector, shape (d,).
    """
    raise NotImplementedError("implied_weights")


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
    raise NotImplementedError("representer_check")


def random_fourier_features(X: np.ndarray, D: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Return random Fourier features whose dot products approximate the RBF kernel.

    Z = sqrt(2 / D) * cos(X @ omega.T + phase), where omega has shape (D, d)
    with independent normal entries of variance 2 * gamma, and phase has shape
    (D,) with entries uniform on [0, 2 pi). Draw omega first, then phase. All
    randomness comes from `rng`, and the shapes of omega and phase depend only
    on D and d, so a generator with the same seed gives the same feature map
    for any number of rows.

    Args:
        X: Points, shape (n, d).
        D: Number of random features.
        gamma: The RBF parameter of the kernel being approximated.
        rng: A numpy random generator.

    Returns:
        The feature matrix Z, shape (n, D), with Z @ Z.T approximately equal to
        the RBF Gram matrix of X.
    """
    raise NotImplementedError("random_fourier_features")


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
    raise NotImplementedError("ridge_on_features")


if __name__ == "__main__":
    x = np.array([0.51, 2.37, 3.21, 3.82, 5.93, 6.13])[:, None]
    y = np.array([0.36, 0.73, -0.39, -0.58, -0.30, 0.17])
    gamma, lam = 1.0, 0.1

    K = rbf_kernel(x, x, gamma)
    alpha = kernel_ridge_fit(K, y, lam)
    train_pred = kernel_ridge_predict(K, alpha)
    print("alpha:", np.round(alpha, 3))
    print(f"training RMSE: {np.sqrt(np.mean((train_pred - y) ** 2)):.4f}")
    print("miss equals lam * alpha:", np.allclose(y - train_pred, lam * alpha))

    X1 = np.array([[1.0], [2.0]])
    a = kernel_ridge_fit(X1 @ X1.T, np.array([1.0, 2.0]), 1.0)
    print("linear kernel: alpha =", np.round(a, 4), " w =", implied_weights(X1, a), " (ordinary ridge gives 5/6)")

    # The representer example: K = I_2, one training target y = 2, lam = 1.
    K2, y2 = np.eye(2), np.array([2.0])
    print("J for c = (1, 0):", representer_check(K2, y2, 1.0, np.array([1.0, 0.0])), "(expected 2)")
    print("J for c = (1, 1):", representer_check(K2, y2, 1.0, np.array([1.0, 1.0])), "(expected 3)")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    Kx = rbf_kernel(X, X, 0.5)
    for D in (10, 160, 2560):
        Z = random_fourier_features(X, D, 0.5, np.random.default_rng(1))
        print(f"D={D:>5}: mean absolute error against the exact RBF Gram matrix = {np.abs(Kx - Z @ Z.T).mean():.4f}")

    xs = np.sort(rng.uniform(0, 6, size=40))[:, None]
    ys = np.sin(xs[:, 0]) + 0.1 * rng.normal(size=40)
    x_test = np.linspace(0, 6, 25)[:, None]
    exact = rbf_kernel(x_test, xs, 1.0) @ kernel_ridge_fit(rbf_kernel(xs, xs, 1.0), ys, 0.1)
    Z = random_fourier_features(xs, 2000, 1.0, np.random.default_rng(7))
    Z_test = random_fourier_features(x_test, 2000, 1.0, np.random.default_rng(7))
    w = ridge_on_features(Z, ys, 0.1)
    print(f"largest gap between random-feature ridge and exact kernel ridge: {np.abs(Z_test @ w - exact).max():.4f}")
