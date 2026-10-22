"""Read every direction from the SVD.

The SVD of centred data lists the directions of greatest spread in order, and
each squared singular value over n - 1 is the variance its direction holds. The
covariance table has the same directions, found with an eigendecomposition.
You build both routes and the variance shares.
"""

from __future__ import annotations

import numpy as np


# Written in the chapter "Spread, shadows, and the first direction".
def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    return X - mean, mean


def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA through the SVD of the centred data, without forming the covariance.

    Args:
        X: Data matrix, shape (n, d).
        k: Number of components to keep.

    Returns:
        A tuple `(components, scores, explained_variance)`.
        `components` holds the top k directions as columns, shape (d, k).
        `scores` is the centred data projected onto them, shape (n, k).
        `explained_variance` is `s[:k] ** 2 / (n - 1)` for the singular values
        `s` of the centred data, shape (k,).
    """
    raise NotImplementedError("pca_svd")


def covariance_eigh(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvalues and eigenvectors of the sample covariance, largest first.

    Args:
        X: Data matrix, shape (n, d).
        k: Number of eigenpairs to keep.

    Returns:
        A tuple `(eigenvalues, eigenvectors)`. `eigenvalues` has shape (k,) in
        descending order. `eigenvectors` has shape (d, k), one unit vector per
        column. The covariance divides by `n - 1`.
    """
    raise NotImplementedError("covariance_eigh")


def explained_variance_ratio(X: np.ndarray) -> np.ndarray:
    """The share of the total variance held by each direction.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        An array of length `min(n, d)` in descending order that sums to one.
        When n is at most d, the last share is zero.
    """
    raise NotImplementedError("explained_variance_ratio")


if __name__ == "__main__":
    X = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    components, scores, variances = pca_svd(X, 2)
    print("variances from the SVD:", variances)
    print("first direction:", components[:, 0])
    print("shares:", explained_variance_ratio(X))

    eigenvalues, eigenvectors = covariance_eigh(X, 2)
    print("eigenvalues of the covariance:", eigenvalues)

    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 2
    true_components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.array([3.0, 1.5])
    data = latent @ true_components.T + 0.2 * rng.normal(size=(n, d))
    ratios = explained_variance_ratio(data)
    keep = int(np.searchsorted(np.cumsum(ratios), 0.90) + 1)
    print("shares on six-dimensional data:", np.round(ratios, 3))
    print("components needed for 90 percent:", keep)
