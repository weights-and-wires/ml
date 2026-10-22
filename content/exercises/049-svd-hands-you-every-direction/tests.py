"""Checks for the SVD-directions exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

HAND = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
U1 = np.array([1.0, 1.0]) / np.sqrt(2.0)


def _gapped(seed=0, n=200, d=5):
    """Data whose variances along the axes of a random rotation are 25, 9, 4, 1, 0.25."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.array([5.0, 3.0, 2.0, 1.0, 0.5])[:d]
    Z = rng.normal(size=(n, d)) * scales
    return Z @ Q.T + rng.normal(size=d)


@check("pca_svd gives variances 6 and 2, with first direction (1,1)/sqrt2, on the three points")
def test_pca_hand(sub):
    got = has(sub, "pca_svd")
    components, scores, variances = got(HAND, 2)
    shape(components, (2, 2), label="components")
    shape(scores, (3, 2), label="scores")
    shape(variances, (2,), label="explained_variance")
    close(variances, np.array([6.0, 2.0]), atol=1e-9, label="explained variances")
    almost(abs(float(components[:, 0] @ U1)), 1.0, tol=1e-9, label="|first direction . (1,1)/sqrt2|")
    close(np.abs(scores[:, 0]), np.array([np.sqrt(2.0), np.sqrt(2.0), 2.0 * np.sqrt(2.0)]), atol=1e-9, label="size of the first scores")


@check("pca_svd returns orthonormal components, and scores that are Xc @ components")
def test_pca_random(sub):
    got = has(sub, "pca_svd")
    X = _gapped(seed=1)
    k = 3
    components, scores, variances = got(X, k)
    shape(components, (5, k), label="components")
    shape(scores, (200, k), label="scores")
    close(components.T @ components, np.eye(k), atol=1e-8, label="components orthonormal")
    Xc = X - X.mean(axis=0)
    close(scores, Xc @ components, atol=1e-8, label="scores vs Xc @ components")
    if np.any(np.diff(variances) > 1e-10):
        fail(f"explained_variance must be in descending order, got {variances}")
    close(np.var(scores, axis=0, ddof=1), variances, rtol=1e-8, label="variance of each score column vs explained_variance")


@check("no random direction has more variance than the first component, and the sum of all variances is trace(C)")
def test_first_is_peak(sub):
    got = has(sub, "pca_svd")
    X = _gapped(seed=2)
    Xc = X - X.mean(axis=0)
    components, _, variances = got(X, 5)
    top = float(variances[0])
    rng = np.random.default_rng(3)
    for _ in range(200):
        u = rng.normal(size=5)
        u /= np.linalg.norm(u)
        v = float(np.var(Xc @ u, ddof=1))
        if v > top + 1e-9:
            fail(f"a random direction has variance {v:.6f}, more than the first component's {top:.6f}")
    almost(float(np.sum(variances)), float(np.trace(np.cov(Xc, rowvar=False))), tol=1e-8 * top, label="sum of variances vs trace of the covariance")


@check("covariance_eigh returns eigenvalues 6 and 2 for the three points, largest first")
def test_eigh_hand(sub):
    got = has(sub, "covariance_eigh")
    eigenvalues, eigenvectors = got(HAND, 2)
    shape(eigenvalues, (2,), label="eigenvalues")
    shape(eigenvectors, (2, 2), label="eigenvectors")
    close(eigenvalues, np.array([6.0, 2.0]), atol=1e-9, label="eigenvalues")
    almost(abs(float(eigenvectors[:, 0] @ U1)), 1.0, tol=1e-9, label="|first eigenvector . (1,1)/sqrt2|")


@check("covariance_eigh eigenvectors satisfy C v = lambda v and are orthonormal")
def test_eigh_equation(sub):
    got = has(sub, "covariance_eigh")
    X = _gapped(seed=4)
    C = np.cov(X, rowvar=False)
    k = 3
    eigenvalues, V = got(X, k)
    shape(V, (5, k), label="eigenvectors")
    if np.any(np.diff(eigenvalues) > 1e-10):
        fail(f"eigenvalues must be in descending order, got {eigenvalues}")
    close(C @ V, V * eigenvalues, atol=1e-8, label="C v vs lambda v")
    close(V.T @ V, np.eye(k), atol=1e-8, label="eigenvectors orthonormal")


@check("the SVD route and the eigh route agree up to sign on data with a clear gap")
def test_routes_agree(sub):
    svd = has(sub, "pca_svd")
    eig = has(sub, "covariance_eigh")
    X = _gapped(seed=5)
    k = 3
    components, _, variances = svd(X, k)
    eigenvalues, eigenvectors = eig(X, k)
    close(variances, eigenvalues, rtol=1e-8, label="variances vs eigenvalues")
    for i in range(k):
        almost(abs(float(components[:, i] @ eigenvectors[:, i])), 1.0, tol=1e-6, label=f"|component {i} . eigenvector {i}|")


@check("explained_variance_ratio gives 0.75 and 0.25 on the three points and sums to one")
def test_ratio_hand(sub):
    got = has(sub, "explained_variance_ratio")
    close(np.asarray(got(HAND)), np.array([0.75, 0.25]), atol=1e-9, label="shares of the three points")
    X = _gapped(seed=6)
    r = np.asarray(got(X))
    shape(r, (5,), label="shares")
    almost(float(np.sum(r)), 1.0, tol=1e-9, label="sum of shares")
    if np.any(np.diff(r) > 1e-12):
        fail(f"shares must be in descending order, got {r}")
    Xc = X - X.mean(axis=0)
    close(r, np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1] / np.trace(np.cov(Xc, rowvar=False)), atol=1e-9, label="shares vs eigenvalue over trace")


@check("explained_variance_ratio keeps min(n, d) shares, and the last is zero when n is at most d")
def test_ratio_wide(sub):
    got = has(sub, "explained_variance_ratio")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4, 6))
    r = np.asarray(got(X))
    shape(r, (4,), label="shares for 4 points in 6 dimensions")
    almost(float(r[-1]), 0.0, tol=1e-9, label="the last share")
    almost(float(np.sum(r)), 1.0, tol=1e-9, label="sum of shares")
