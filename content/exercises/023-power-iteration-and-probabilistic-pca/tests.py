"""Checks for the power iteration and probabilistic PCA exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

A21 = np.diag([2.0, 1.0])
TINY = np.array([[2.0, 0.0], [-2.0, 0.0]])
W1 = np.array([[1.0], [0.0]])


def _spd(eigs, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T, Q


def _data(seed=0, n=250, d=6, k=2):
    rng = np.random.default_rng(seed)
    comps = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.linspace(3.0, 1.5, k)
    return latent @ comps.T + 0.2 * rng.normal(size=(n, d))


@check("rayleigh_quotient is the eigenvalue at an eigenvector and a weighted average elsewhere")
def test_rayleigh(sub):
    got = has(sub, "rayleigh_quotient")
    almost(float(got(A21, np.array([1.0, 0.0]))), 2.0, tol=1e-12, label="quotient at (1, 0)")
    almost(float(got(A21, np.array([0.0, 1.0]))), 1.0, tol=1e-12, label="quotient at (0, 1)")
    almost(float(got(A21, np.array([1.0, 1.0]) / np.sqrt(2.0))), 1.5, tol=1e-12, label="quotient at (1,1)/sqrt2")
    almost(float(got(A21, np.array([0.8944271910, 0.4472135955]))), 1.8, tol=1e-8, label="quotient after one step")
    almost(float(got(A21, np.array([3.0, 3.0]))), 1.5, tol=1e-12, label="quotient at a vector of length 3 sqrt2")
    A, _ = _spd([5.0, 3.0, 2.0, 1.0, 0.5], seed=1)
    rng = np.random.default_rng(2)
    for _ in range(20):
        v = rng.normal(size=5)
        q = float(got(A, v))
        if not (0.5 - 1e-9 <= q <= 5.0 + 1e-9):
            fail(f"a quotient must lie between the smallest and largest eigenvalue, got {q}")


@check("power_iteration reproduces the hand trace: (0.8944, 0.4472) and 1.8, then (0.9701, 0.2425) and 33/17")
def test_trace(sub):
    got = has(sub, "power_iteration")
    v, eigenvalue = got(A21, 1)
    shape(v, (2,), label="v")
    close(v, np.array([2.0, 1.0]) / np.sqrt(5.0), atol=1e-9, label="v after one step")
    almost(float(eigenvalue), 1.8, tol=1e-9, label="Rayleigh quotient after one step")
    v, eigenvalue = got(A21, 2)
    close(v, np.array([4.0, 1.0]) / np.sqrt(17.0), atol=1e-9, label="v after two steps")
    almost(float(eigenvalue), 33.0 / 17.0, tol=1e-9, label="Rayleigh quotient after two steps")


@check("power_iteration rescales every step, so thousands of steps do not overflow, and a large negative eigenvalue wins with its sign")
def test_no_overflow(sub):
    got = has(sub, "power_iteration")
    v, eigenvalue = got(A21, 5000)
    if not (np.all(np.isfinite(v)) and np.isfinite(eigenvalue)):
        fail(f"got non-finite values after 5000 steps: v = {v}, eigenvalue = {eigenvalue}. Rescale inside the loop.")
    almost(float(np.linalg.norm(v)), 1.0, tol=1e-9, label="length of v")
    almost(float(eigenvalue), 2.0, tol=1e-9, label="eigenvalue after 5000 steps")
    v, eigenvalue = got(np.diag([-3.0, 1.0, 0.5]), 60)
    almost(abs(float(v[0])), 1.0, tol=1e-6, label="weight of v on the eigenvalue -3 direction")
    almost(float(eigenvalue), -3.0, tol=1e-6, label="eigenvalue")


@check("the direction error shrinks by lambda2/lambda1 per step, and the eigenvalue error by its square")
def test_rates(sub):
    got = has(sub, "power_iteration")
    eigs = np.array([2.0, 1.6, 1.2, 0.9, 0.6, 0.3])
    A, Q = _spd(eigs, seed=42)
    u1 = Q[:, 0]
    ratio = eigs[1] / eigs[0]

    def errors(t):
        v, lam = got(A, t)
        return np.sqrt(max(0.0, 1.0 - (v @ u1) ** 2)), abs(eigs[0] - lam)

    a10, r10 = errors(10)
    a30, r30 = errors(30)
    if min(a10, a30, r10, r30) < 1e-14:
        fail("an error underflowed; the matrix gap in the check is too large for your result")
    direction_rate = (a30 / a10) ** (1.0 / 20)
    eigenvalue_rate = (r30 / r10) ** (1.0 / 20)
    almost(direction_rate, ratio, tol=0.05 * ratio, label=f"direction error rate versus lambda2/lambda1 = {ratio}")
    almost(eigenvalue_rate, ratio**2, tol=0.05 * ratio**2, label=f"eigenvalue error rate versus the ratio squared = {ratio**2}")


@check("deflate leaves the second eigenvalue on top, and power iteration plus deflation finds the first two PCA components")
def test_deflate(sub):
    got = has(sub, "deflate")
    eigs = np.array([4.0, 2.5, 1.5, 1.0])
    A, Q = _spd(eigs, seed=3)
    D = np.asarray(got(A, Q[:, 0], 4.0))
    shape(D, (4, 4), label="deflated matrix")
    close(D @ Q[:, 0], np.zeros(4), atol=1e-9, label="deflated matrix times the found direction")
    close(np.sort(np.linalg.eigvalsh(D))[::-1], np.array([2.5, 1.5, 1.0, 0.0]), atol=1e-9, label="eigenvalues of the deflated matrix")
    C = np.array([[4.0, 2.0], [2.0, 4.0]])
    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    close(np.asarray(got(C, u, 6.0)), np.array([[1.0, -1.0], [-1.0, 1.0]]), atol=1e-9, label="deflated three-point covariance")
    power = has(sub, "power_iteration")
    pca_svd = has(sub, "pca_svd")
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    X = (rng.normal(size=(300, 6)) * np.array([4.0, 2.5, 1.0, 0.7, 0.4, 0.2])) @ Q.T
    C = np.cov(X, rowvar=False)
    v1, l1 = power(C, 300)
    v2, l2 = power(got(C, v1, l1), 300)
    components, _, variances = pca_svd(X, 2)
    almost(float(l1), float(variances[0]), tol=1e-6, label="first eigenvalue vs first explained variance")
    almost(float(l2), float(variances[1]), tol=1e-5, label="second eigenvalue vs second explained variance")
    almost(abs(float(v1 @ components[:, 0])), 1.0, tol=1e-6, label="|v1 . first component|")
    almost(abs(float(v2 @ components[:, 1])), 1.0, tol=1e-5, label="|v2 . second component|")


@check("ppca_e_step gives Ez = (1, -1) and Ezz = 3 on the two-point example, and matches a point-by-point posterior")
def test_e_step(sub):
    got = has(sub, "ppca_e_step")
    Ez, Ezz = got(TINY, W1, 1.0)
    shape(Ez, (2, 1), label="Ez")
    shape(Ezz, (1, 1), label="Ezz_sum")
    close(Ez, np.array([[1.0], [-1.0]]), atol=1e-10, label="Ez")
    close(Ezz, np.array([[3.0]]), atol=1e-10, label="Ezz_sum")
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


@check("ppca_m_step gives W = 4/3 and sigma2 = 2/3 on the two-point example, and clips sigma2 at 1e-8 when the fit is exact")
def test_m_step(sub):
    got = has(sub, "ppca_m_step")
    W, sigma2 = got(TINY, np.array([[1.0], [-1.0]]), np.array([[3.0]]))
    shape(W, (2, 1), label="W_new")
    close(W, np.array([[4.0 / 3.0], [0.0]]), atol=1e-10, label="W_new")
    almost(float(sigma2), 2.0 / 3.0, tol=1e-10, label="sigma2_new")
    W, sigma2 = got(TINY, np.array([[1.0], [-1.0]]), np.array([[2.0]]))
    close(W, np.array([[2.0], [0.0]]), atol=1e-10, label="W_new")
    if not (1e-8 - 1e-15 <= float(sigma2) <= 1e-6):
        fail(f"sigma2 should be clipped to about 1e-8 when the leftover is zero, got {sigma2}")


@check("ppca_loglik gives -6.37 and -5.80 on the two-point example, and equals the sum of Gaussian log densities")
def test_loglik(sub):
    got = has(sub, "ppca_loglik")
    before = float(got(TINY, W1, 1.0))
    after = float(got(TINY, np.array([[4.0 / 3.0], [0.0]]), 2.0 / 3.0))
    want_before = -(2 * np.log(2 * np.pi) + np.log(2.0) + 2.0)
    want_after = -(2 * np.log(2 * np.pi) + np.log(44.0 / 27.0) + 18.0 / 11.0)
    almost(before, want_before, tol=1e-9, label="log-likelihood before")
    almost(after, want_after, tol=1e-9, label="log-likelihood after")
    if not (after > before):
        fail("the log-likelihood should rise from the first to the second value")
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


@check("the EM loop's log-likelihood never decreases, the fit lands on the PCA plane, and sigma2 is the average of the dropped variances")
def test_em(sub):
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
