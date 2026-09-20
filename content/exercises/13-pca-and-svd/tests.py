"""Checks for the PCA and SVD exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def _low_rank_data(n=250, d=6, k=2, seed=0):
    rng = np.random.default_rng(seed)
    components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    scale = np.linspace(3.0, 1.0, k)
    latent = rng.normal(size=(n, k)) * scale
    X = latent @ components.T + 0.2 * rng.normal(size=(n, d))
    return X


@check("center subtracts the column mean, and adding it back recovers X")
def test_center(sub):
    got = has(sub, "center")
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(40, 4))
    Xc, mean = got(X)
    shape(Xc, X.shape, label="centred data")
    shape(mean, (4,), label="mean vector")
    close(Xc.mean(axis=0), np.zeros(4), atol=1e-10, label="column means of the centred data")
    close(Xc + mean, X, atol=1e-10, label="undoing the centring")


@check("pca_svd returns orthonormal components with scores equal to Xc @ components")
def test_pca_svd_basic(sub):
    got = has(sub, "pca_svd")
    center = has(sub, "center")
    X = _low_rank_data(seed=1)
    k = 2
    components, scores, explained_variance = got(X, k)

    shape(components, (X.shape[1], k), label="components")
    shape(scores, (X.shape[0], k), label="scores")
    shape(explained_variance, (k,), label="explained_variance")
    close(components.T @ components, np.eye(k), atol=1e-8, label="components orthonormal")

    Xc, _ = center(X)
    close(scores, Xc @ components, atol=1e-8, rtol=1e-8, label="scores vs Xc @ components")
    if np.any(np.diff(explained_variance) > 1e-10):
        fail(f"explained_variance must be in descending order, got {explained_variance}")


@check("pca_svd agrees with a direct eigendecomposition of the sample covariance")
def test_pca_svd_matches_eigh(sub):
    got = has(sub, "pca_svd")
    center = has(sub, "center")
    X = _low_rank_data(seed=2)
    n = X.shape[0]
    k = 3

    Xc, _ = center(X)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    components, _, explained_variance = got(X, k)
    close(explained_variance, eigvals[:k], rtol=1e-6, label="explained variance vs eigenvalues")

    for i in range(k):
        sign = 1.0 if np.dot(components[:, i], eigvecs[:, i]) >= 0 else -1.0
        close(
            components[:, i],
            sign * eigvecs[:, i],
            atol=1e-6,
            label=f"component {i} vs the matching eigenvector, up to sign",
        )


@check("reconstruction error matches the Eckart-Young bound sum_{i>k} sigma_i^2 exactly")
def test_eckart_young(sub):
    pca_svd = has(sub, "pca_svd")
    reconstruct = has(sub, "reconstruct")
    center = has(sub, "center")
    X = _low_rank_data(seed=3, n=200, d=8, k=3)
    k = 2

    Xc, mean = center(X)
    s_full = np.linalg.svd(Xc, compute_uv=False)
    bound = float(np.sum(s_full[k:] ** 2))

    components, scores, _ = pca_svd(X, k)
    Xhat = reconstruct(mean, components, scores)
    error = float(np.sum((X - Xhat) ** 2))

    almost(error, bound, tol=1e-6 * max(1.0, bound), label="reconstruction error vs the Eckart-Young bound")


@check("explained_variance_ratio sums to one across the full rank")
def test_explained_variance_ratio(sub):
    got = has(sub, "explained_variance_ratio")
    center = has(sub, "center")
    X = _low_rank_data(seed=4, n=100, d=5)

    ratios = got(X)
    if np.any(ratios < -1e-10):
        fail(f"explained variance ratios must be nonnegative, got {ratios}")
    almost(float(np.sum(ratios)), 1.0, tol=1e-8, label="sum of explained variance ratios")

    Xc, _ = center(X)
    s = np.linalg.svd(Xc, compute_uv=False)
    want = s**2 / np.sum(s**2)
    close(np.sort(ratios)[::-1], np.sort(want)[::-1], rtol=1e-6, label="ratios vs direct computation")


@check("power_iteration converges to the top eigenvector at the rate the gap predicts")
def test_power_iteration(sub):
    got = has(sub, "power_iteration")
    rng = np.random.default_rng(42)
    d = 6
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.array([2.0, 1.6, 1.2, 0.9, 0.6, 0.3])
    A = Q @ np.diag(eigs) @ Q.T
    u1 = Q[:, 0]
    gap = eigs[1] / eigs[0]

    def alignment_error(t):
        v, eigenvalue = got(A, t)
        almost(np.linalg.norm(v), 1.0, tol=1e-6, label=f"unit norm at t={t}")
        return np.sqrt(max(0.0, 1.0 - (v @ u1) ** 2)), eigenvalue

    err10, _ = alignment_error(10)
    err30, eigenvalue30 = alignment_error(30)

    if err10 < 1e-12 or err30 < 1e-15:
        fail("alignment error underflowed; the test matrix's gap needs to be less extreme")

    empirical_rate = (err30 / err10) ** (1.0 / (30 - 10))
    almost(
        empirical_rate,
        gap,
        tol=0.05 * gap,
        label=f"empirical convergence rate over t=10..30 versus the eigenvalue gap {gap}",
    )
    almost(eigenvalue30, eigs[0], tol=1e-4, label="Rayleigh quotient at t=30 versus the top eigenvalue")


@check("ppca_em's log-likelihood never decreases across EM iterations")
def test_ppca_monotone(sub):
    got = has(sub, "ppca_em")
    X = _low_rank_data(seed=5, n=150, d=6, k=2)

    _, _, history = got(X, k=2, iters=60)
    if len(history) != 60:
        fail(f"expected a log-likelihood history of length 60, got {len(history)}")

    for m in range(len(history) - 1):
        if history[m + 1] < history[m] - 1e-6:
            fail(
                f"log-likelihood decreased from {history[m]:.6f} at iteration {m + 1} "
                f"to {history[m + 1]:.6f} at iteration {m + 2}"
            )
    if history[-1] <= history[0]:
        fail(
            f"log-likelihood should increase substantially over 60 iterations, went "
            f"from {history[0]:.4f} to {history[-1]:.4f}"
        )


@check("ppca_em's fitted subspace converges to the PCA subspace")
def test_ppca_subspace(sub):
    ppca_em = has(sub, "ppca_em")
    pca_svd = has(sub, "pca_svd")
    X = _low_rank_data(seed=6, n=250, d=6, k=2)
    k = 2

    components, _, _ = pca_svd(X, k)
    projector_pca = components @ components.T

    W, _, _ = ppca_em(X, k=k, iters=200)
    Q, _ = np.linalg.qr(W)
    projector_ppca = Q @ Q.T

    diff = float(np.linalg.norm(projector_pca - projector_ppca))
    if diff > 0.02:
        fail(
            "after 200 EM iterations, the subspace spanned by the fitted W should "
            f"match the top-k PCA subspace closely, but the projector difference "
            f"(Frobenius norm) is {diff:.4f}"
        )
