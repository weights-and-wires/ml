"""Checks for the probabilistic PCA exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

TINY = np.array([[2.0, 0.0], [-2.0, 0.0]])
W1 = np.array([[1.0], [0.0]])


def _data(seed=0, n=250, d=6, k=2):
    rng = np.random.default_rng(seed)
    comps = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.linspace(3.0, 1.5, k)
    return latent @ comps.T + 0.2 * rng.normal(size=(n, d))


@check("ppca_e_step gives Ez = (1, -1) and Ezz = 3 on the two-point example")
def test_e_hand(sub):
    got = has(sub, "ppca_e_step")
    Ez, Ezz = got(TINY, W1, 1.0)
    shape(Ez, (2, 1), label="Ez")
    shape(Ezz, (1, 1), label="Ezz_sum")
    close(Ez, np.array([[1.0], [-1.0]]), atol=1e-10, label="Ez")
    close(Ezz, np.array([[3.0]]), atol=1e-10, label="Ezz_sum")


@check("ppca_e_step matches the posterior mean and second moment computed point by point")
def test_e_general(sub):
    got = has(sub, "ppca_e_step")
    rng = np.random.default_rng(1)
    n, d, k = 12, 5, 2
    Xc = rng.normal(size=(n, d))
    Xc -= Xc.mean(axis=0)
    W = rng.normal(size=(d, k))
    sigma2 = 0.7
    M_inv = np.linalg.inv(W.T @ W + sigma2 * np.eye(k))
    Ez_want = np.array([M_inv @ W.T @ x for x in Xc])
    Ezz_want = sum(sigma2 * M_inv + np.outer(z, z) for z in Ez_want)
    Ez, Ezz = got(Xc, W, sigma2)
    close(Ez, Ez_want, atol=1e-9, label="Ez against M^-1 W^T x per point")
    close(Ezz, Ezz_want, atol=1e-9, label="Ezz_sum against the per-point sum")


@check("ppca_m_step gives W = 4/3 and sigma2 = 2/3 on the two-point example")
def test_m_hand(sub):
    got = has(sub, "ppca_m_step")
    W, sigma2 = got(TINY, np.array([[1.0], [-1.0]]), np.array([[3.0]]))
    shape(W, (2, 1), label="W_new")
    close(W, np.array([[4.0 / 3.0], [0.0]]), atol=1e-10, label="W_new")
    almost(float(sigma2), 2.0 / 3.0, tol=1e-10, label="sigma2_new")


@check("ppca_m_step clips sigma2 at 1e-8 when the fit is exact")
def test_m_clip(sub):
    got = has(sub, "ppca_m_step")
    W, sigma2 = got(TINY, np.array([[1.0], [-1.0]]), np.array([[2.0]]))
    close(W, np.array([[2.0], [0.0]]), atol=1e-10, label="W_new")
    if not (1e-8 - 1e-15 <= float(sigma2) <= 1e-6):
        fail(f"sigma2 should be clipped to about 1e-8 when the leftover is zero, got {sigma2}")


@check("ppca_loglik gives -6.37 before and -5.80 after one round on the two-point example")
def test_loglik_hand(sub):
    got = has(sub, "ppca_loglik")
    before = float(got(TINY, W1, 1.0))
    after = float(got(TINY, np.array([[4.0 / 3.0], [0.0]]), 2.0 / 3.0))
    want_before = -(2 * np.log(2 * np.pi) + np.log(2.0) + 2.0)
    want_after = -(2 * np.log(2 * np.pi) + np.log(44.0 / 27.0) + 18.0 / 11.0)
    almost(before, want_before, tol=1e-9, label="log-likelihood before")
    almost(after, want_after, tol=1e-9, label="log-likelihood after")
    if not (after > before):
        fail("the log-likelihood should rise from the first to the second value")


@check("ppca_loglik equals the sum of Gaussian log densities computed directly")
def test_loglik_density(sub):
    got = has(sub, "ppca_loglik")
    rng = np.random.default_rng(2)
    n, d, k = 9, 3, 1
    Xc = rng.normal(size=(n, d))
    Xc -= Xc.mean(axis=0)
    W = rng.normal(size=(d, k))
    sigma2 = 0.4
    C = W @ W.T + sigma2 * np.eye(d)
    Ci = np.linalg.inv(C)
    det = np.linalg.det(C)
    total = sum(-0.5 * (d * np.log(2 * np.pi) + np.log(det) + x @ Ci @ x) for x in Xc)
    almost(float(got(Xc, W, sigma2)), float(total), tol=1e-8, label="log-likelihood against the direct density")


@check("the EM loop's log-likelihood never decreases")
def test_monotone(sub):
    em = has(sub, "ppca_em")
    X = _data(seed=5, n=150)
    _, _, history = em(X, k=2, iters=60)
    if len(history) != 60:
        fail(f"expected a history of length 60, got {len(history)}")
    for m in range(len(history) - 1):
        if history[m + 1] < history[m] - 1e-6:
            fail(f"log-likelihood fell from {history[m]:.6f} at round {m + 1} to {history[m + 1]:.6f} at round {m + 2}")
    if history[-1] <= history[0]:
        fail(f"the log-likelihood should rise over 60 rounds, went from {history[0]:.4f} to {history[-1]:.4f}")


@check("the EM fit lands on the PCA plane, and sigma2 is the average of the dropped variances")
def test_matches_pca(sub):
    em = has(sub, "ppca_em")
    pca = has(sub, "pca_svd")
    X = _data(seed=6, n=250)
    k = 2
    components, _, _ = pca(X, k)
    W, sigma2, _ = em(X, k=k, iters=300)
    Q, _ = np.linalg.qr(W)
    diff = float(np.linalg.norm(Q @ Q.T - components @ components.T))
    if diff > 0.02:
        fail(f"the plane spanned by W should match the top-{k} PCA plane, but the projector difference is {diff:.4f}")
    Xc = X - X.mean(axis=0)
    eigs = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / len(X)))[::-1]
    want = float(np.mean(eigs[k:]))
    almost(float(sigma2), want, tol=0.01 * want, label="sigma2 vs the mean of the dropped variances (divisor n)")
