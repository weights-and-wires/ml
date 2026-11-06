"""Checks for the principal-directions exercise."""

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


U1 = np.array([1.0, 1.0]) / np.sqrt(2.0)


def _gapped(seed=0, n=200, d=5):
    """Data whose variances along the axes of a random rotation are 25, 9, 4, 1, 0.25."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.array([5.0, 3.0, 2.0, 1.0, 0.5])[:d]
    Z = rng.normal(size=(n, d)) * scales
    return Z @ Q.T + rng.normal(size=d)


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


@check("shadow_split gives kept 12 and lost 4 along u, and kept 4 and lost 12 along w")
def test_split_hand(sub):
    got = has(sub, "shadow_split")
    Xc = HAND - np.array([2.0, 2.0])
    kept, lost = got(Xc, U)
    almost(float(kept), 12.0, tol=1e-9, label="kept along u")
    almost(float(lost), 4.0, tol=1e-9, label="lost along u")
    kept, lost = got(Xc, W)
    almost(float(kept), 4.0, tol=1e-9, label="kept along w")
    almost(float(lost), 12.0, tol=1e-9, label="lost along w")


@check("kept plus lost is the total squared length, and the angle that keeps the most loses the least")
def test_split_pythagoras(sub):
    got = has(sub, "shadow_split")
    Xc = _cloud(seed=3)
    rng = np.random.default_rng(4)
    total = float(np.sum(Xc**2))
    for _ in range(4):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        kept, lost = got(Xc, u)
        almost(float(kept) + float(lost), total, tol=1e-8 * total, label="kept + lost")
        # distance from each point to the line, found without reusing the scores
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


@check("covariance_eigh gives eigenvalues 6 and 2 on the three points, and its eigenvectors satisfy C v = lambda v")
def test_eigh(sub):
    got = has(sub, "covariance_eigh")
    eigenvalues, eigenvectors = got(HAND, 2)
    shape(eigenvalues, (2,), label="eigenvalues")
    shape(eigenvectors, (2, 2), label="eigenvectors")
    close(eigenvalues, np.array([6.0, 2.0]), atol=1e-9, label="eigenvalues")
    almost(abs(float(eigenvectors[:, 0] @ U1)), 1.0, tol=1e-9, label="|first eigenvector . (1,1)/sqrt2|")
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
