"""Reference solution for the PCA and SVD exercise."""

import numpy as np


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    return X - mean, mean


def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc, _ = center(X)
    n = X.shape[0]
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    components = vt[:k].T
    scores = Xc @ components
    explained_variance = s[:k] ** 2 / (n - 1)
    return components, scores, explained_variance


def reconstruct(mean: np.ndarray, components: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return mean + scores @ components.T


def explained_variance_ratio(X: np.ndarray) -> np.ndarray:
    Xc, _ = center(X)
    s = np.linalg.svd(Xc, compute_uv=False)
    variances = s**2
    return variances / variances.sum()


def power_iteration(A: np.ndarray, iters: int) -> tuple[np.ndarray, float]:
    d = A.shape[0]
    v = np.ones(d) / np.sqrt(d)
    for _ in range(iters):
        v = A @ v
        v = v / np.linalg.norm(v)
    eigenvalue = float(v @ A @ v)
    return v, eigenvalue


def ppca_em(X: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, float, list[float]]:
    Xc, _ = center(X)
    n, d = Xc.shape

    rng = np.random.default_rng(0)
    W = rng.normal(scale=0.1, size=(d, k))
    sigma2 = 1.0

    S = (Xc.T @ Xc) / n
    history: list[float] = []

    for _ in range(iters):
        M = W.T @ W + sigma2 * np.eye(k)
        M_inv = np.linalg.inv(M)

        Ez = Xc @ W @ M_inv  # (n, k), M symmetric so M_inv.T == M_inv
        Ezz_sum = n * sigma2 * M_inv + Ez.T @ Ez  # (k, k)

        W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz_sum)

        term1 = float(np.sum(Xc**2))
        term2 = 2.0 * float(np.sum(Ez * (Xc @ W_new)))
        term3 = float(np.trace(Ezz_sum @ (W_new.T @ W_new)))
        sigma2_new = max((term1 - term2 + term3) / (n * d), 1e-8)

        W, sigma2 = W_new, sigma2_new

        C = W @ W.T + sigma2 * np.eye(d)
        _, logdet = np.linalg.slogdet(C)
        loglik = -0.5 * n * (d * np.log(2 * np.pi) + logdet + np.trace(np.linalg.solve(C, S)))
        history.append(float(loglik))

    return W, sigma2, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 2

    true_components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.array([3.0, 1.5])
    X = latent @ true_components.T + 0.2 * rng.normal(size=(n, d))

    components, scores, explained_variance = pca_svd(X, k)
    print(f"explained variance (top {k}): {explained_variance}")

    mean = center(X)[1]
    Xhat = reconstruct(mean, components, scores)
    print(f"reconstruction error: {np.sum((X - Xhat) ** 2):.4f}")

    ratios = explained_variance_ratio(X)
    print(f"explained variance ratio sums to {ratios.sum():.6f}")

    cov = np.cov(X, rowvar=False)
    v, eigenvalue = power_iteration(cov, iters=50)
    print(f"power iteration top eigenvalue: {eigenvalue:.4f}")

    W, sigma2, history = ppca_em(X, k, iters=100)
    print(f"ppca sigma2: {sigma2:.4f}, log-likelihood: {history[0]:.2f} -> {history[-1]:.2f}")
