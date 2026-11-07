"""Checks for the principal components exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

U = np.array([1.0, 1.0]) / np.sqrt(2.0)
W = np.array([1.0, -1.0]) / np.sqrt(2.0)
HAND = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])


def _cloud(seed=0, n=60, d=4):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    X = rng.normal(size=(n, d)) @ A + rng.normal(size=d) * 5.0
    return X - X.mean(axis=0)


def _gapped(seed=0, n=200, d=5):
    """Data whose variances along the axes of a random rotation are 25, 9, 4, 1, 0.25."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.array([5.0, 3.0, 2.0, 1.0, 0.5])[:d]
    Z = rng.normal(size=(n, d)) * scales
    return Z @ Q.T + rng.normal(size=d)


def _data(seed=0, n=250, d=6, k=2):
    rng = np.random.default_rng(seed)
    components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.linspace(3.0, 1.5, k)
    return latent @ components.T + 0.2 * rng.normal(size=(n, d))


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
    Xc3, mean3 = got(HAND)
    close(mean3, np.array([2.0, 2.0]), atol=1e-12, label="the middle of the three-point cloud")


@check("variance_along gives 6 and 2 on the three points, matches u @ C @ u, and drives the sweep")
def test_variance(sub):
    got = has(sub, "variance_along")
    Xc = HAND - np.array([2.0, 2.0])
    almost(float(got(Xc, U)), 6.0, tol=1e-9, label="variance along u")
    almost(float(got(Xc, W)), 2.0, tol=1e-9, label="variance along w")
    almost(float(got(Xc, np.array([1.0, 0.0]))), 4.0, tol=1e-9, label="variance along the x axis")
    Xr = _cloud(seed=1)
    rng = np.random.default_rng(2)
    C = np.cov(Xr, rowvar=False)
    for _ in range(4):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        want = float(np.var(Xr @ u, ddof=1))
        almost(float(got(Xr, u)), want, tol=1e-8 * max(1.0, want), label="variance of the scores")
        almost(float(got(Xr, u)), float(u @ C @ u), tol=1e-8 * max(1.0, want), label="u @ C @ u")
    sweep = has(sub, "sweep_directions")
    angles = np.radians(np.arange(0, 180, 5))
    out = np.asarray(sweep(Xc, angles))
    close(out, 4.0 + 2.0 * np.sin(2.0 * angles), atol=1e-9, label="variance against angle")
    almost(float(np.degrees(angles[np.argmax(out)])), 45.0, tol=1e-9, label="best angle in degrees")


@check("shadow_split gives kept 12 and lost 4 along u, kept plus lost is the total, and most kept means least lost")
def test_shadow_split(sub):
    got = has(sub, "shadow_split")
    Xc = HAND - np.array([2.0, 2.0])
    kept, lost = got(Xc, U)
    almost(float(kept), 12.0, tol=1e-9, label="kept along u")
    almost(float(lost), 4.0, tol=1e-9, label="lost along u")
    kept, lost = got(Xc, W)
    almost(float(kept), 4.0, tol=1e-9, label="kept along w")
    almost(float(lost), 12.0, tol=1e-9, label="lost along w")
    Xc = _cloud(seed=3)
    rng = np.random.default_rng(4)
    total = float(np.sum(Xc**2))
    for _ in range(4):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        kept, lost = got(Xc, u)
        almost(float(kept) + float(lost), total, tol=1e-8 * total, label="kept + lost")
        proj = Xc @ np.outer(u, u)
        almost(float(lost), float(np.sum((Xc - proj) ** 2)), tol=1e-8 * total, label="lost vs distance to the line")
    rng = np.random.default_rng(5)
    Xc = rng.normal(size=(80, 2)) @ np.array([[2.0, 1.0], [0.0, 0.6]])
    Xc -= Xc.mean(axis=0)
    angles = np.radians(np.arange(0, 180, 3))
    pairs = [got(Xc, np.array([np.cos(a), np.sin(a)])) for a in angles]
    kept = np.array([p[0] for p in pairs])
    lost = np.array([p[1] for p in pairs])
    if int(np.argmax(kept)) != int(np.argmin(lost)):
        fail(f"most kept is at angle index {int(np.argmax(kept))}, but least lost is at {int(np.argmin(lost))}")


@check("pca_svd gives variances 6 and 2 on the three points, and orthonormal components with no direction beating the first")
def test_pca_svd(sub):
    got = has(sub, "pca_svd")
    components, scores, variances = got(HAND, 2)
    shape(components, (2, 2), label="components")
    shape(scores, (3, 2), label="scores")
    shape(variances, (2,), label="explained_variance")
    close(variances, np.array([6.0, 2.0]), atol=1e-9, label="explained variances")
    almost(abs(float(components[:, 0] @ U)), 1.0, tol=1e-9, label="|first direction . (1,1)/sqrt2|")
    close(np.abs(scores[:, 0]), np.array([np.sqrt(2.0), np.sqrt(2.0), 2.0 * np.sqrt(2.0)]), atol=1e-9, label="size of the first scores")
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
    _, _, all_variances = got(X, 5)
    top = float(all_variances[0])
    rng = np.random.default_rng(3)
    for _ in range(200):
        u = rng.normal(size=5)
        u /= np.linalg.norm(u)
        v = float(np.var(Xc @ u, ddof=1))
        if v > top + 1e-9:
            fail(f"a random direction has variance {v:.6f}, more than the first component's {top:.6f}")
    almost(float(np.sum(all_variances)), float(np.trace(np.cov(Xc, rowvar=False))), tol=1e-8 * top, label="sum of variances vs trace of the covariance")


@check("covariance_eigh gives eigenvalues 6 and 2, satisfies C v = lambda v, and agrees with pca_svd up to sign")
def test_eigh(sub):
    got = has(sub, "covariance_eigh")
    eigenvalues, eigenvectors = got(HAND, 2)
    shape(eigenvalues, (2,), label="eigenvalues")
    shape(eigenvectors, (2, 2), label="eigenvectors")
    close(eigenvalues, np.array([6.0, 2.0]), atol=1e-9, label="eigenvalues")
    almost(abs(float(eigenvectors[:, 0] @ U)), 1.0, tol=1e-9, label="|first eigenvector . (1,1)/sqrt2|")
    X = _gapped(seed=4)
    C = np.cov(X, rowvar=False)
    k = 3
    eigenvalues, V = got(X, k)
    shape(V, (5, k), label="eigenvectors")
    if np.any(np.diff(eigenvalues) > 1e-10):
        fail(f"eigenvalues must be in descending order, got {eigenvalues}")
    close(C @ V, V * eigenvalues, atol=1e-8, label="C v vs lambda v")
    close(V.T @ V, np.eye(k), atol=1e-8, label="eigenvectors orthonormal")
    svd = has(sub, "pca_svd")
    components, _, variances = svd(X, k)
    close(variances, eigenvalues, rtol=1e-8, label="variances vs eigenvalues")
    for i in range(k):
        almost(abs(float(components[:, i] @ V[:, i])), 1.0, tol=1e-6, label=f"|component {i} . eigenvector {i}|")


@check("explained_variance_ratio gives 0.75 and 0.25 on the three points, sums to one, and keeps min(n, d) shares")
def test_ratio(sub):
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
    rng = np.random.default_rng(7)
    wide = np.asarray(got(rng.normal(size=(4, 6))))
    shape(wide, (4,), label="shares for 4 points in 6 dimensions")
    almost(float(wide[-1]), 0.0, tol=1e-9, label="the last share")
    almost(float(np.sum(wide)), 1.0, tol=1e-9, label="sum of shares")


@check("reconstruct rebuilds the three points as (1,1), (1,1), (4,4), and gives X back exactly with every direction")
def test_reconstruct(sub):
    got = has(sub, "reconstruct")
    u = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
    scores = np.array([[-np.sqrt(2.0)], [-np.sqrt(2.0)], [2.0 * np.sqrt(2.0)]])
    out = np.asarray(got(np.array([2.0, 2.0]), u, scores))
    shape(out, (3, 2), label="rebuilt points")
    close(out, np.array([[1.0, 1.0], [1.0, 1.0], [4.0, 4.0]]), atol=1e-9, label="rebuilt points")
    pca_svd = has(sub, "pca_svd")
    X = _data(seed=1, n=60, d=5)
    components, scores, _ = pca_svd(X, 5)
    close(np.asarray(got(X.mean(axis=0), components, scores)), X, atol=1e-8, label="rebuild with k = d")


@check("reconstruction_error is 4 for k = 1, equals the dropped squared singular values at every k, and no random sheet beats it")
def test_reconstruction_error(sub):
    got = has(sub, "reconstruction_error")
    almost(float(got(HAND, 1)), 4.0, tol=1e-9, label="error with k = 1")
    almost(float(got(HAND, 2)), 0.0, tol=1e-9, label="error with k = 2")
    almost(float(got(HAND, 0)), 16.0, tol=1e-9, label="error with k = 0")
    X = _data(seed=2, n=200, d=8, k=3)
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    for k in range(0, 9):
        want = float(np.sum(s[k:] ** 2))
        almost(float(got(X, k)), want, tol=1e-7 * max(1.0, want) + 1e-7, label=f"error at k = {k}")
    rec = has(sub, "reconstruct")
    X = _data(seed=3, n=150, d=6)
    mean = X.mean(axis=0)
    Xc = X - mean
    rng = np.random.default_rng(4)
    for k in (1, 2, 3):
        best = float(got(X, k))
        for _ in range(60):
            Q, _ = np.linalg.qr(rng.normal(size=(6, k)))
            X_hat = rec(mean, Q, Xc @ Q)
            other = float(np.sum((X - X_hat) ** 2))
            if other < best - 1e-9:
                fail(f"a random {k}-direction sheet lost {other:.4f}, less than the top {k} directions' {best:.4f}")
    ratio = has(sub, "explained_variance_ratio")
    X = _data(seed=5)
    total = float(np.sum((X - X.mean(axis=0)) ** 2))
    cumulative = np.cumsum(ratio(X))
    for k in range(1, 7):
        almost(1.0 - float(got(X, k)) / total, float(cumulative[k - 1]), tol=1e-8, label=f"kept share at k = {k}")


@check("standardise gives mean 0 and standard deviation 1, leaves a constant column at 0, and removes the effect of units")
def test_standardise(sub):
    got = has(sub, "standardise")
    rng = np.random.default_rng(6)
    X = rng.normal(loc=[1.0, -4.0, 10.0], scale=[0.5, 20.0, 3.0], size=(80, 3))
    Z = np.asarray(got(X))
    shape(Z, X.shape, label="standardised table")
    close(Z.mean(axis=0), np.zeros(3), atol=1e-10, label="column means")
    close(Z.std(axis=0, ddof=1), np.ones(3), atol=1e-10, label="column standard deviations")
    flat = np.column_stack([X[:, 0], np.full(80, 7.0)])
    out = np.asarray(got(flat))
    if not np.all(np.isfinite(out)):
        fail("a constant column produced nan or inf; leave it at 0 instead of dividing by 0")
    close(out[:, 1], np.zeros(80), atol=1e-12, label="constant column")
    pca_svd = has(sub, "pca_svd")
    rng = np.random.default_rng(7)
    base = rng.normal(size=(100, 1))
    A = np.column_stack([base[:, 0] + 0.3 * rng.normal(size=100), base[:, 0] + 0.3 * rng.normal(size=100), rng.normal(size=100)])
    scaled = A.copy()
    scaled[:, 1] *= 1000.0
    raw_a = pca_svd(A, 1)[0][:, 0]
    raw_b = pca_svd(scaled, 1)[0][:, 0]
    if abs(float(raw_a @ raw_b)) > 0.99:
        fail("setup error: changing units should change the raw first direction")
    a = pca_svd(np.asarray(got(A)), 1)[0][:, 0]
    b = pca_svd(np.asarray(got(scaled)), 1)[0][:, 0]
    almost(abs(float(a @ b)), 1.0, tol=1e-8, label="|first direction (kg) . first direction (g)| after standardising")
