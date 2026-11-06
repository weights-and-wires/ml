"""Reference solution: rebuild from k directions, and measure what you lost.

Keeping the top k directions rebuilds each row as the mean plus a mix of k
directions. The total squared error of that rebuild is the sum of the squared
singular values you dropped, and no other k directions do better. You build the
rebuild, measure its error, and standardise columns so that units stop mattering.
"""

from __future__ import annotations

import numpy as np


# Written in the chapter "Principal directions: where the data spreads".
def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    return X - mean, mean


# Written in the chapter "Principal directions: where the data spreads".
def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc, _ = center(X)
    n = X.shape[0]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].T
    scores = Xc @ components
    return components, scores, s[:k] ** 2 / (n - 1)


# Written in the chapter "Principal directions: where the data spreads".
def explained_variance_ratio(X: np.ndarray) -> np.ndarray:
    Xc, _ = center(X)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s**2 / np.sum(s**2)


def reconstruct(mean: np.ndarray, components: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rebuild data from its scores on k directions.

    Args:
        mean: The mean removed before projecting, shape (d,).
        components: The kept directions as columns, shape (d, k).
        scores: Coordinates on those directions, shape (n, k).

    Returns:
        The rebuilt data, shape (n, d): the mean plus `scores @ components.T`.
    """
    return mean + scores @ components.T


def reconstruction_error(X: np.ndarray, k: int) -> float:
    """Total squared error of rebuilding X from its top k directions.

    Run `pca_svd`, rebuild with `reconstruct`, and add up the squared
    differences from `X`. Do not shortcut with the singular values.

    Args:
        X: Data matrix, shape (n, d).
        k: Number of directions to keep, from 0 to d.

    Returns:
        The sum over all entries of `(X - X_hat) ** 2`, as a float.
    """
    components, scores, _ = pca_svd(X, k)
    X_hat = reconstruct(X.mean(axis=0), components, scores)
    return float(np.sum((X - X_hat) ** 2))


def standardise(X: np.ndarray) -> np.ndarray:
    """Centre each column and divide it by its standard deviation.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        An array of shape (n, d) whose columns have mean 0 and standard
        deviation 1 (with `ddof=1`). A column whose standard deviation is 0 is
        left at 0 instead of being divided.
    """
    Xc, _ = center(X)
    std = X.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    return Xc / std


if __name__ == "__main__":
    X = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    components, scores, _ = pca_svd(X, 1)
    print("rebuilt from one direction:\n", reconstruct(X.mean(axis=0), components, scores))
    print("error with k = 1:", reconstruction_error(X, 1))

    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 2
    true_components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.array([3.0, 1.5])
    data = latent @ true_components.T + 0.2 * rng.normal(size=(n, d))
    s = np.linalg.svd(data - data.mean(axis=0), compute_uv=False)
    for keep in range(d + 1):
        print(f"k = {keep}: error {reconstruction_error(data, keep):9.3f}, dropped s^2 {np.sum(s[keep:] ** 2):9.3f}")

    grams = np.array([[2.0, 0.0], [0.0, 200.0], [4.0, 400.0]])
    print("first direction, raw:", pca_svd(grams, 1)[0][:, 0])
    print("first direction, standardised:", pca_svd(standardise(grams), 1)[0][:, 0])
