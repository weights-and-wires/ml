"""Checks for the power-iteration exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

A21 = np.diag([2.0, 1.0])


def _spd(eigs, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T, Q


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


@check("power_iteration reproduces the hand trace: step 1 gives (0.8944, 0.4472) and 1.8")
def test_trace_step1(sub):
    got = has(sub, "power_iteration")
    v, eigenvalue = got(A21, 1)
    shape(v, (2,), label="v")
    close(v, np.array([2.0, 1.0]) / np.sqrt(5.0), atol=1e-9, label="v after one step")
    almost(float(eigenvalue), 1.8, tol=1e-9, label="Rayleigh quotient after one step")


@check("power_iteration reproduces the hand trace: step 2 gives (0.9701, 0.2425) and 33/17")
def test_trace_step2(sub):
    got = has(sub, "power_iteration")
    v, eigenvalue = got(A21, 2)
    close(v, np.array([4.0, 1.0]) / np.sqrt(17.0), atol=1e-9, label="v after two steps")
    almost(float(eigenvalue), 33.0 / 17.0, tol=1e-9, label="Rayleigh quotient after two steps")


@check("power_iteration rescales every step, so thousands of steps do not overflow")
def test_no_overflow(sub):
    got = has(sub, "power_iteration")
    v, eigenvalue = got(A21, 5000)
    if not (np.all(np.isfinite(v)) and np.isfinite(eigenvalue)):
        fail(f"got non-finite values after 5000 steps: v = {v}, eigenvalue = {eigenvalue}. Rescale inside the loop.")
    almost(float(np.linalg.norm(v)), 1.0, tol=1e-9, label="length of v")
    almost(float(eigenvalue), 2.0, tol=1e-9, label="eigenvalue after 5000 steps")


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


@check("a large negative eigenvalue wins by size, and the quotient reports it with its sign")
def test_negative(sub):
    got = has(sub, "power_iteration")
    A = np.diag([-3.0, 1.0, 0.5])
    v, eigenvalue = got(A, 60)
    almost(abs(float(v[0])), 1.0, tol=1e-6, label="weight of v on the eigenvalue -3 direction")
    almost(float(eigenvalue), -3.0, tol=1e-6, label="eigenvalue")


@check("deflate maps the found direction to zero and leaves the second eigenvalue on top")
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


@check("power iteration plus deflation on a covariance matrix finds the first two PCA components")
def test_matches_pca(sub):
    power = has(sub, "power_iteration")
    deflate = has(sub, "deflate")
    pca_svd = has(sub, "pca_svd")
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    X = (rng.normal(size=(300, 6)) * np.array([4.0, 2.5, 1.0, 0.7, 0.4, 0.2])) @ Q.T
    C = np.cov(X, rowvar=False)
    v1, l1 = power(C, 300)
    v2, l2 = power(deflate(C, v1, l1), 300)
    components, _, variances = pca_svd(X, 2)
    almost(float(l1), float(variances[0]), tol=1e-6, label="first eigenvalue vs first explained variance")
    almost(float(l2), float(variances[1]), tol=1e-5, label="second eigenvalue vs second explained variance")
    almost(abs(float(v1 @ components[:, 0])), 1.0, tol=1e-6, label="|v1 . first component|")
    almost(abs(float(v2 @ components[:, 1])), 1.0, tol=1e-5, label="|v2 . second component|")
