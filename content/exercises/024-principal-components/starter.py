"""Principal components: compress and know what you lost.

A line through the middle of a cloud turns each point into one score, the
length of its shadow on the line. The variance of the scores says how spread
out the shadows are, and each point's squared length splits into a kept part
(shadow squared) and a lost part (gap squared). The SVD of the centred data
lists every direction in order of spread, and the covariance table has the
same directions. Keeping the top k directions rebuilds each row as the mean
plus a mix of k directions. The total squared error of that rebuild is the sum
of the squared singular values you dropped, and no other k directions do
better. You write ten functions, in this order:

1. `center`, 2. `variance_along`, 3. `shadow_split`, 4. `pca_svd`,
5. `covariance_eigh`, 6. `explained_variance_ratio`, 7. `reconstruct`,
8. `reconstruction_error`, 9. `standardise`, 10. `whiten`.

`sweep_directions` is already written, because it only calls `variance_along`.
"""

from __future__ import annotations

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


def variance_along(Xc: np.ndarray, u: np.ndarray) -> float:
    """Variance of the scores of centred data along a unit vector.

    Args:
        Xc: Centred data, shape (n, d).
        u: A unit vector, shape (d,).

    Returns:
        The sum of squared scores `Xc @ u`, divided by `n - 1`, as a float.
    """
    raise NotImplementedError("variance_along")


def shadow_split(Xc: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Split the total squared length of centred data into kept and lost parts.

    Args:
        Xc: Centred data, shape (n, d).
        u: A unit vector, shape (d,).

    Returns:
        A tuple `(kept, lost)`. `kept` is the sum over points of the squared
        shadow on `u`. `lost` is the sum over points of the squared length of
        what is left after you subtract each shadow, `x - (x @ u) * u`.
    """
    raise NotImplementedError("shadow_split")


# Given: it only calls your variance_along once per angle.
def sweep_directions(Xc: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Variance along the unit vector (cos a, sin a), for each angle a in radians.

    Args:
        Xc: Centred data with two columns, shape (n, 2).
        angles: Angles in radians, shape (m,).

    Returns:
        The variances, shape (m,).
    """
    return np.array([variance_along(Xc, np.array([np.cos(a), np.sin(a)])) for a in angles])


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


def reconstruct(mean: np.ndarray, components: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rebuild data from its scores on k directions.

    Args:
        mean: The mean removed before projecting, shape (d,).
        components: The kept directions as columns, shape (d, k).
        scores: Coordinates on those directions, shape (n, k).

    Returns:
        The rebuilt data, shape (n, d): the mean plus `scores @ components.T`.
    """
    raise NotImplementedError("reconstruct")


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
    raise NotImplementedError("reconstruction_error")


def standardise(X: np.ndarray) -> np.ndarray:
    """Centre each column and divide it by its standard deviation.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        An array of shape (n, d) whose columns have mean 0 and standard
        deviation 1 (with `ddof=1`). A column whose standard deviation is 0 is
        left at 0 instead of being divided.
    """
    raise NotImplementedError("standardise")


def whiten(X: np.ndarray, eps: float = 1e-5, method: str = "pca") -> tuple[np.ndarray, np.ndarray]:
    """Rotate and rescale X so its covariance is the identity.

    Centre X, take the eigenvalues `lam` and eigenvectors `U` of its sample
    covariance (divide by `n - 1`), and build a whitening matrix `W` with
    `scale = 1 / sqrt(lam + eps)`:

    - `"pca"`: `W = diag(scale) @ U.T`.
    - `"zca"`: `W = U @ diag(scale) @ U.T`.

    Args:
        X: Data matrix, shape (n, d).
        eps: Added to every eigenvalue before the square root, so a direction
            with almost no variance is not blown up.
        method: `"pca"` or `"zca"`. Any other value raises `ValueError`.

    Returns:
        A tuple `(Z, W)`. `Z` is the centred data times `W.T`, shape (n, d).
        `W` is the whitening matrix, shape (d, d).
    """
    raise NotImplementedError("whiten")



if __name__ == "__main__":
    X = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    Xc, mean = center(X)
    print("middle of the cloud:", mean)

    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    w = np.array([1.0, -1.0]) / np.sqrt(2.0)
    print(f"variance along u: {variance_along(Xc, u):.4f}, along w: {variance_along(Xc, w):.4f}")

    kept, lost = shadow_split(Xc, u)
    print(f"line u keeps {kept:.4f} and loses {lost:.4f}")

    angles = np.radians(np.arange(0, 180, 5))
    best = angles[np.argmax(sweep_directions(Xc, angles))]
    print(f"best angle: {np.degrees(best):.0f} degrees")

    components, scores, variances = pca_svd(X, 2)
    print("variances from the SVD:", variances)
    print("first direction:", components[:, 0])
    print("shares:", explained_variance_ratio(X))

    eigenvalues, eigenvectors = covariance_eigh(X, 2)
    print("eigenvalues of the covariance:", eigenvalues)

    components, scores, _ = pca_svd(X, 1)
    print("rebuilt from one direction:\n", reconstruct(X.mean(axis=0), components, scores))
    print("error with k = 1:", reconstruction_error(X, 1))

    rng = np.random.default_rng(0)
    n, d, k = 300, 6, 2
    true_components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.array([3.0, 1.5])
    data = latent @ true_components.T + 0.2 * rng.normal(size=(n, d))
    ratios = explained_variance_ratio(data)
    keep = int(np.searchsorted(np.cumsum(ratios), 0.90) + 1)
    print("shares on six-dimensional data:", np.round(ratios, 3))
    print("components needed for 90 percent:", keep)
    s = np.linalg.svd(data - data.mean(axis=0), compute_uv=False)
    for kk in range(d + 1):
        print(f"k = {kk}: error {reconstruction_error(data, kk):9.3f}, dropped s^2 {np.sum(s[kk:] ** 2):9.3f}")

    grams = np.array([[2.0, 0.0], [0.0, 200.0], [4.0, 400.0]])
    print("first direction, raw:", pca_svd(grams, 1)[0][:, 0])
    print("first direction, standardised:", pca_svd(standardise(grams), 1)[0][:, 0])

    Z, W = whiten(X, eps=0.0, method="pca")
    print("PCA-whitened points:\n", np.round(Z, 2))
    Z, W = whiten(X, eps=0.0, method="zca")
    print("ZCA matrix:\n", np.round(W, 3))
    print("ZCA-whitened points:\n", np.round(Z, 2))
