"""Checks for the compress-and-know-what-you-lost exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

HAND = np.array([[2.0, 0.0], [0.0, 2.0], [4.0, 4.0]])


def _data(seed=0, n=250, d=6, k=2):
    rng = np.random.default_rng(seed)
    components = np.linalg.qr(rng.normal(size=(d, k)))[0]
    latent = rng.normal(size=(n, k)) * np.linspace(3.0, 1.5, k)
    return latent @ components.T + 0.2 * rng.normal(size=(n, d))


@check("reconstruct rebuilds the three points as (1,1), (1,1), (4,4) from one direction")
def test_reconstruct_hand(sub):
    got = has(sub, "reconstruct")
    u = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
    scores = np.array([[-np.sqrt(2.0)], [-np.sqrt(2.0)], [2.0 * np.sqrt(2.0)]])
    out = np.asarray(got(np.array([2.0, 2.0]), u, scores))
    shape(out, (3, 2), label="rebuilt points")
    close(out, np.array([[1.0, 1.0], [1.0, 1.0], [4.0, 4.0]]), atol=1e-9, label="rebuilt points")


@check("reconstruct with every direction gives X back exactly")
def test_reconstruct_full(sub):
    got = has(sub, "reconstruct")
    pca_svd = has(sub, "pca_svd")
    X = _data(seed=1, n=60, d=5)
    components, scores, _ = pca_svd(X, 5)
    close(np.asarray(got(X.mean(axis=0), components, scores)), X, atol=1e-8, label="rebuild with k = d")


@check("reconstruction_error is 4 for k = 1, 0 for k = 2, and 16 for k = 0 on the three points")
def test_error_hand(sub):
    got = has(sub, "reconstruction_error")
    almost(float(got(HAND, 1)), 4.0, tol=1e-9, label="error with k = 1")
    almost(float(got(HAND, 2)), 0.0, tol=1e-9, label="error with k = 2")
    almost(float(got(HAND, 0)), 16.0, tol=1e-9, label="error with k = 0")


@check("reconstruction_error equals the sum of the dropped squared singular values at every k")
def test_eckart_young_formula(sub):
    got = has(sub, "reconstruction_error")
    X = _data(seed=2, n=200, d=8, k=3)
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    for k in range(0, 9):
        want = float(np.sum(s[k:] ** 2))
        almost(float(got(X, k)), want, tol=1e-7 * max(1.0, want) + 1e-7, label=f"error at k = {k}")


@check("no other sheet of k directions loses less than the top k")
def test_no_better_sheet(sub):
    got = has(sub, "reconstruction_error")
    rec = has(sub, "reconstruct")
    X = _data(seed=3, n=150, d=6)
    mean = X.mean(axis=0)
    Xc = X - mean
    rng = np.random.default_rng(4)
    for k in (1, 2, 3):
        best = float(got(X, k))
        for _ in range(100):
            Q, _ = np.linalg.qr(rng.normal(size=(6, k)))
            X_hat = rec(mean, Q, Xc @ Q)
            other = float(np.sum((X - X_hat) ** 2))
            if other < best - 1e-9:
                fail(f"a random {k}-direction sheet lost {other:.4f}, less than the top {k} directions' {best:.4f}")


@check("one minus error over total equals the cumulative explained variance ratio")
def test_error_vs_ratio(sub):
    got = has(sub, "reconstruction_error")
    ratio = has(sub, "explained_variance_ratio")
    X = _data(seed=5)
    Xc = X - X.mean(axis=0)
    total = float(np.sum(Xc**2))
    cumulative = np.cumsum(ratio(X))
    for k in range(1, 7):
        almost(1.0 - float(got(X, k)) / total, float(cumulative[k - 1]), tol=1e-8, label=f"kept share at k = {k}")


@check("standardise gives columns with mean 0 and standard deviation 1, and leaves a constant column at 0")
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


@check("standardising removes the effect of changing a column's units")
def test_units(sub):
    std = has(sub, "standardise")
    pca_svd = has(sub, "pca_svd")
    rng = np.random.default_rng(7)
    base = rng.normal(size=(100, 1))
    X = np.column_stack([base[:, 0] + 0.3 * rng.normal(size=100), base[:, 0] + 0.3 * rng.normal(size=100), rng.normal(size=100)])
    scaled = X.copy()
    scaled[:, 1] *= 1000.0
    raw_a = pca_svd(X, 1)[0][:, 0]
    raw_b = pca_svd(scaled, 1)[0][:, 0]
    if abs(float(raw_a @ raw_b)) > 0.99:
        fail("setup error: changing units should change the raw first direction")
    a = pca_svd(np.asarray(std(X)), 1)[0][:, 0]
    b = pca_svd(np.asarray(std(scaled)), 1)[0][:, 0]
    almost(abs(float(a @ b)), 1.0, tol=1e-8, label="|first direction (kg) . first direction (g)| after standardising")
