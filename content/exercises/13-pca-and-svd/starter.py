"""PCA through the SVD, power iteration, and probabilistic PCA by EM.

Fill in the six functions below. `center` and `reconstruct` are the glue
between the other four: `pca_svd` and `ppca_em` both operate on centred data,
and `reconstruct` undoes the centring to compare against the original `X`.
"""

import numpy as np


def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the column mean from X.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        A tuple `(Xc, mean)`. `Xc` is `X` with its column mean subtracted,
        shape (n, d). `mean` is the column mean removed, shape (d,).
    """
    raise NotImplementedError("center")


def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA through the SVD of the centred data, without forming the covariance.

    Args:
        X: Data matrix, shape (n, d).
        k: Number of components to keep.

    Returns:
        A tuple `(components, scores, explained_variance)`.
        `components` holds the top k right singular vectors as columns,
        shape (d, k), the principal axes. `scores` is the data projected onto
        those axes, shape (n, k), equal to `center(X)[0] @ components`.
        `explained_variance` is the variance captured by each of the k
        components, shape (k,), equal to `s[:k] ** 2 / (n - 1)` for the
        singular values `s` of the centred data.
    """
    raise NotImplementedError("pca_svd")


def reconstruct(mean: np.ndarray, components: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Invert a PCA projection back into the original space.

    Args:
        mean: The mean removed before projecting, shape (d,).
        components: Principal axes, shape (d, k).
        scores: Projected coordinates, shape (n, k).

    Returns:
        The reconstructed data, shape (n, d): `mean + scores @ components.T`.
    """
    raise NotImplementedError("reconstruct")


def explained_variance_ratio(X: np.ndarray) -> np.ndarray:
    """The full-rank explained variance ratio of the centred data.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        An array of length `min(n - 1, d)` (the rank of the centred data in
        the generic case), holding each component's share of total variance,
        in descending order, summing to one.
    """
    raise NotImplementedError("explained_variance_ratio")


def power_iteration(A: np.ndarray, iters: int) -> tuple[np.ndarray, float]:
    """Find the top eigenvector of a symmetric matrix by power iteration.

    Start from the fixed vector `np.ones(d) / sqrt(d)` and repeat
    `v = A @ v; v = v / norm(v)` for `iters` steps.

    Args:
        A: A symmetric matrix, shape (d, d).
        iters: Number of iterations to run.

    Returns:
        A tuple `(v, eigenvalue)`. `v` is the unit-norm estimate of the top
        eigenvector, shape (d,). `eigenvalue` is the Rayleigh quotient
        `v @ A @ v`, an estimate of the corresponding eigenvalue.
    """
    raise NotImplementedError("power_iteration")


def ppca_em(X: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, float, list[float]]:
    """Fit probabilistic PCA by expectation-maximisation.

    Initialise W with `np.random.default_rng(0).normal(scale=0.1, size=(d, k))`
    and sigma2 with 1.0. At each iteration, run the E-step and M-step derived
    in the lesson, then record the marginal log-likelihood of the centred
    data under the updated (W, sigma2).

    Args:
        X: Data matrix, shape (n, d).
        k: Latent dimension.
        iters: Number of EM iterations.

    Returns:
        A tuple `(W, sigma2, loglik_history)`. `W` is the fitted loading
        matrix, shape (d, k). `sigma2` is the fitted noise variance, a float.
        `loglik_history` is a list of length `iters`, the marginal
        log-likelihood after each iteration.
    """
    raise NotImplementedError("ppca_em")


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
