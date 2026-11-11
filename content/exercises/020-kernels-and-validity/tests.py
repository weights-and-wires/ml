"""Checks for the kernels and validity exercise."""

import numpy as np

from mlcheck import check, close, fail, shape


@check("rbf_kernel matches a hand-built example, is symmetric with a unit diagonal, and never exceeds 1")
def test_rbf(sub):
    X1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    X2 = np.array([[0.0, 1.0]])
    gamma = 0.5
    # Squared distances: (0,0)-(0,1) is 1, and (1,0)-(0,1) is 2.
    want = np.exp(-gamma * np.array([[1.0], [2.0]]))
    got = sub.rbf_kernel(X1, X2, gamma)
    shape(got, (2, 1), label="rbf_kernel output")
    close(got, want, atol=1e-10, label="RBF kernel values")

    rng = np.random.default_rng(0)
    # Large coordinates make the expanded squared distance lose digits, so a
    # missing clip at zero pushes some entries slightly above 1.
    X = 1000.0 * rng.normal(size=(12, 3))
    K = sub.rbf_kernel(X, X, 1e-3)
    shape(K, (12, 12), label="Gram matrix")
    close(K, K.T, atol=1e-9, label="Gram matrix against its transpose")
    close(np.diag(K), np.ones(12), atol=1e-6, label="diagonal")
    if K.max() > 1.0:
        fail(f"an RBF similarity must not exceed 1, but the largest entry is {K.max():.12f}; clip squared distances at zero")


@check("A larger gamma makes the RBF bump narrower, so off-diagonal similarity falls")
def test_rbf_gamma(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 2))
    n = len(X)
    means = []
    for gamma in (0.1, 1.0, 10.0):
        K = sub.rbf_kernel(X, X, gamma)
        means.append((K.sum() - np.trace(K)) / (n * (n - 1)))
    if not (means[0] > means[1] > means[2]):
        fail(f"mean off-diagonal similarity should fall as gamma grows, got {means} for gamma 0.1, 1, 10")


@check("polynomial_kernel gives 36 for (1,2) and (3,1) at degree 2, and degree 1 with coef0 0 is X1 @ X2.T")
def test_polynomial(sub):
    X1 = np.array([[1.0, 2.0], [0.0, 1.0]])
    X2 = np.array([[3.0, 1.0]])
    got = sub.polynomial_kernel(X1, X2, 2, 1.0)
    shape(got, (2, 1), label="polynomial_kernel output")
    # Dot products: 1*3 + 2*1 = 5 and 0*3 + 1*1 = 1.
    close(got, np.array([[36.0], [4.0]]), atol=1e-10, label="degree-2 polynomial kernel values")
    cubic = sub.polynomial_kernel(X1, X2, 3, 2.0)
    close(cubic, np.array([[7.0**3], [3.0**3]]), atol=1e-10, label="degree-3 polynomial kernel with coef0 2")

    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 4))
    B = rng.normal(size=(9, 4))
    lin = sub.polynomial_kernel(A, B, 1, 0.0)
    shape(lin, (6, 9), label="cross-kernel matrix")
    close(lin, A @ B.T, atol=1e-10, label="linear kernel")


@check("The kernel trick: poly_features_degree2 has the six documented columns, and its dot products equal the degree-2 kernel")
def test_features_and_trick(sub):
    X = np.array([[1.0, 2.0], [3.0, 1.0]])
    phi = sub.poly_features_degree2(X)
    shape(phi, (2, 6), label="feature matrix")
    r = np.sqrt(2.0)
    want = np.array([[1.0, r * 1.0, r * 2.0, 1.0, r * 2.0, 4.0], [1.0, r * 3.0, r * 1.0, 9.0, r * 3.0, 1.0]])
    close(phi, want, atol=1e-10, label="features of (1,2) and (3,1)")

    rng = np.random.default_rng(3)
    X1 = rng.normal(size=(7, 2))
    X2 = rng.normal(size=(5, 2))
    explicit = sub.poly_features_degree2(X1) @ sub.poly_features_degree2(X2).T
    shortcut = sub.polynomial_kernel(X1, X2, 2, 1.0)
    shape(explicit, (7, 5), label="explicit-feature Gram matrix")
    close(explicit, shortcut, atol=1e-10, rtol=1e-10, label="phi(X1) phi(X2)^T against (X1 X2^T + 1)^2")


@check("min_eigenvalue finds the smallest eigenvalue of hand-built matrices")
def test_min_eigenvalue(sub):
    close(sub.min_eigenvalue(np.diag([3.0, 1.0, 2.0])), 1.0, atol=1e-10, label="diagonal matrix")
    # Eigenvalues of [[1, 2], [2, 1]] are 3 and -1.
    got = sub.min_eigenvalue(np.array([[1.0, 2.0], [2.0, 1.0]]))
    close(got, -1.0, atol=1e-10, label="invented table")
    if not isinstance(got, float):
        fail(f"min_eigenvalue should return a Python float, got {type(got).__name__}")


@check("combination_squared_length gives -2 for the invented table and equals ||Phi^T v||^2 when K = Phi Phi^T")
def test_combination(sub):
    v = np.array([1.0, -1.0])
    invented = np.array([[1.0, 2.0], [2.0, 1.0]])
    # 1*1 + 1*1 - 2*2 = -2
    close(sub.combination_squared_length(invented, v), -2.0, atol=1e-12, label="invented table")
    # 1 + 1 - 2*0 = 2
    close(sub.combination_squared_length(np.eye(2), v), 2.0, atol=1e-12, label="identity")

    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(6, 4))
    K = Phi @ Phi.T
    for _ in range(5):
        w = rng.normal(size=6)
        got = sub.combination_squared_length(K, w)
        want = float(np.sum((Phi.T @ w) ** 2))
        close(got, want, atol=1e-9, rtol=1e-9, label="v^T K v against ||Phi^T v||^2")
        if got < 0:
            fail(f"a Gram matrix of real features gave a negative squared length {got}")


@check("is_psd accepts RBF and polynomial Gram matrices, including one with a zero eigenvalue")
def test_is_psd_accepts(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    if not sub.is_psd(sub.rbf_kernel(X, X, 1.0)):
        fail("an RBF Gram matrix is valid, but is_psd rejected it")
    if not sub.is_psd(sub.polynomial_kernel(X, X, 2, 1.0)):
        fail("a polynomial Gram matrix with coef0 = 1 is valid, but is_psd rejected it")
    # coef0 = 0, degree 2, in 3-D has only 6 features, so 8 points give zero eigenvalues up to rounding.
    if not sub.is_psd(sub.polynomial_kernel(X, X, 2, 0.0)):
        fail("a Gram matrix whose smallest eigenvalue is zero up to rounding is valid; allow eigenvalues down to -tol")
    if not sub.is_psd(np.array([[1.0, 1.0], [1.0, 1.0]])):
        fail("duplicated points give a zero eigenvalue, which is allowed")


@check("is_psd rejects the invented table, an asymmetric or non-square matrix, and a polynomial Gram matrix with coef0 = -2")
def test_is_psd_rejects(sub):
    if sub.is_psd(np.array([[1.0, 2.0], [2.0, 1.0]])):
        fail("the invented table has eigenvalues 3 and -1, so it is not PSD, but is_psd accepted it")
    if sub.is_psd(np.ones((2, 3))):
        fail("a non-square matrix cannot be a Gram matrix, but is_psd accepted it")
    # Both eigenvalues of this matrix are 1, so only the asymmetry should reject it.
    if sub.is_psd(np.array([[1.0, 2.0], [0.0, 1.0]])):
        fail("an asymmetric matrix cannot be a Gram matrix, but is_psd accepted it")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    K = sub.polynomial_kernel(X, X, 2, -2.0)
    if np.linalg.eigvalsh(K)[0] > -1.0:
        fail("test construction error: expected a clearly negative eigenvalue")
    if sub.is_psd(K):
        fail("the degree-2 polynomial Gram matrix with coef0 = -2 has a clearly negative eigenvalue, but is_psd accepted it")

    # The upper entry is off by 1e-3, far above tol but far below the relative slack np.allclose allows at this size.
    big = np.array([[1e6, 1e6 + 1e-3], [1e6, 1e6]])
    if sub.is_psd(big, tol=1e-8):
        fail("entries differ from their transpose by 1e-3, more than tol=1e-8; use np.allclose(K, K.T, atol=tol, rtol=0.0)")
    tiny_noise = np.array([[1.0, 0.5 + 1e-10], [0.5, 1.0]])
    if not sub.is_psd(tiny_noise, tol=1e-8):
        fail("a mismatch of 1e-10 is below tol=1e-8, so the matrix should count as symmetric")


@check("The eigenvector of a negative eigenvalue is a witness with negative squared length")
def test_witness(sub):
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 6))
    indefinite = A + A.T
    for K in (np.array([[1.0, 2.0], [2.0, 1.0]]), indefinite):
        lo = sub.min_eigenvalue(K)
        if lo >= -1e-6:
            fail(f"test construction error: expected a negative eigenvalue, got {lo}")
        _, vecs = np.linalg.eigh(K)
        v = vecs[:, 0]
        got = sub.combination_squared_length(K, v)
        close(got, lo, atol=1e-9, label="v^T K v for a unit eigenvector equals its eigenvalue")
        if got >= 0:
            fail(f"expected a negative squared length, got {got}")


def _pca_scores(X):
    """PCA scores of the centred points, largest singular value first.

    The reference lives in this file so the check does not import another lesson.
    """
    centred = X - X.mean(axis=0)
    U, S, _Vt = np.linalg.svd(centred, full_matrices=False)
    keep = S > 1e-8
    return U[:, keep] * S[keep]


def _kernel_pca_scores(Kc):
    evals, evecs = np.linalg.eigh(Kc)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    keep = evals > 1e-8
    return evecs[:, keep] * np.sqrt(evals[keep])


def _align_signs(got, want):
    out = np.array(got, dtype=float, copy=True)
    for j in range(out.shape[1]):
        if np.dot(out[:, j], want[:, j]) < 0:
            out[:, j] *= -1
    return out


@check("centre_gram matches H K H, and kernel PCA on a linear kernel matches the PCA scores")
def test_centre_gram(sub):
    # Catches a centre_gram that subtracts the row mean only (K minus the row
    # mean, which is K @ H rather than H @ K @ H). On the matrix below that
    # leaves the top-left entry at 1/3 instead of 5/9, and the scores disagree.
    K = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    want = np.array([
        [5.0 / 9.0, -4.0 / 9.0, -1.0 / 9.0],
        [-4.0 / 9.0, 5.0 / 9.0, -1.0 / 9.0],
        [-1.0 / 9.0, -1.0 / 9.0, 2.0 / 9.0],
    ])
    got = sub.centre_gram(K)
    shape(got, (3, 3), label="centred Gram matrix")
    close(got, want, atol=1e-10, label="centred three-point Gram matrix")

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pca = _pca_scores(X)
    kpca = _kernel_pca_scores(got)
    if kpca.shape != pca.shape:
        fail(
            f"kernel PCA scores: expected shape {pca.shape}, actual shape {kpca.shape}. "
            f"expected {pca}, actual {kpca}"
        )
    close(_align_signs(kpca, pca), pca, atol=1e-8, label="kernel PCA scores against PCA scores of the centred points")

    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(8, 3))
    centred = Xr - Xr.mean(axis=0)
    got_r = sub.centre_gram(Xr @ Xr.T)
    close(got_r, centred @ centred.T, atol=1e-8, label="centred linear kernel against the Gram matrix of the centred points")
    pca_r = _pca_scores(Xr)
    kpca_r = _kernel_pca_scores(got_r)
    if kpca_r.shape != pca_r.shape:
        fail(
            f"random-point kernel PCA scores: expected shape {pca_r.shape}, actual shape {kpca_r.shape}. "
            f"expected {pca_r}, actual {kpca_r}"
        )
    close(
        _align_signs(kpca_r, pca_r),
        pca_r,
        atol=1e-7,
        label="kernel PCA scores against PCA scores on eight random points",
    )
