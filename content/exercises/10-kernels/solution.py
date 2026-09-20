"""Reference solution for the kernels and representer theorem exercise."""

import numpy as np


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    sq_dist = (
        np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
    )
    np.maximum(sq_dist, 0.0, out=sq_dist)
    return np.exp(-gamma * sq_dist)


def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, coef0: float = 1.0) -> np.ndarray:
    return (X1 @ X2.T + coef0) ** degree


def is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    if K.shape[0] != K.shape[1]:
        return False
    if not np.allclose(K, K.T, atol=tol):
        return False
    eigenvalues = np.linalg.eigvalsh(K)
    return bool(eigenvalues.min() >= -tol)


def kernel_ridge_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = K.shape[0]
    L = np.linalg.cholesky(K + lam * np.eye(n))
    z = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, z)


def kernel_ridge_predict(K_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return K_test @ alpha


def random_fourier_features(X: np.ndarray, D: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    omega = rng.normal(size=(D, d)) * np.sqrt(2.0 * gamma)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ omega.T + phase[None, :])


def representer_check(K: np.ndarray, y: np.ndarray, lam: float, coeffs: np.ndarray) -> float:
    n = len(y)
    predictions = (K @ coeffs)[:n]
    loss = float(np.sum((y - predictions) ** 2))
    norm_sq = float(coeffs @ K @ coeffs)
    return loss + lam * norm_sq


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 30, 4
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + 0.1 * rng.normal(size=n)
    gamma, lam = 0.5, 0.2

    K = rbf_kernel(X, X, gamma)
    print(f"RBF Gram matrix is PSD: {is_psd(K)}")

    alpha = kernel_ridge_fit(K, y, lam)
    train_pred = kernel_ridge_predict(K, alpha)
    print(f"training RMSE: {np.sqrt(np.mean((train_pred - y) ** 2)):.4f}")

    for D in (50, 500, 5000):
        Z = random_fourier_features(X, D, gamma, np.random.default_rng(1))
        approx = Z @ Z.T
        error = np.abs(K - approx).mean()
        print(f"D={D:>5}: mean absolute error against the exact RBF Gram matrix = {error:.4f}")

    baseline = representer_check(K, y, lam, alpha)
    print(f"regularised objective at the representer-optimal alpha: {baseline:.4f}")
