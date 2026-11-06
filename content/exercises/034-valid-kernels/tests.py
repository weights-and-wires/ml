"""Checks for the valid-kernels exercise."""

import numpy as np

from mlcheck import check, close, fail


@check("min_eigenvalue finds the smallest eigenvalue of hand-built matrices")
def test_min_eigenvalue(sub):
    close(sub.min_eigenvalue(np.diag([3.0, 1.0, 2.0])), 1.0, atol=1e-10, label="diagonal matrix")
    # Eigenvalues of [[1, 2], [2, 1]] are 3 and -1.
    got = sub.min_eigenvalue(np.array([[1.0, 2.0], [2.0, 1.0]]))
    close(got, -1.0, atol=1e-10, label="invented table")
    if not isinstance(got, float):
        fail(f"min_eigenvalue should return a Python float, got {type(got).__name__}")


@check("combination_squared_length gives -2 for the invented table and 2 for the identity")
def test_combination_hand(sub):
    v = np.array([1.0, -1.0])
    invented = np.array([[1.0, 2.0], [2.0, 1.0]])
    # 1*1 + 1*1 - 2*2 = -2
    close(sub.combination_squared_length(invented, v), -2.0, atol=1e-12, label="invented table")
    # 1 + 1 - 2*0 = 2
    close(sub.combination_squared_length(np.eye(2), v), 2.0, atol=1e-12, label="identity")


@check("combination_squared_length equals the squared norm of Phi^T v when K = Phi Phi^T")
def test_combination_features(sub):
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(6, 4))
    K = Phi @ Phi.T
    for _ in range(5):
        v = rng.normal(size=6)
        got = sub.combination_squared_length(K, v)
        want = float(np.sum((Phi.T @ v) ** 2))
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


@check("is_psd rejects the invented table, a non-square matrix, and an asymmetric matrix")
def test_is_psd_rejects_shapes(sub):
    if sub.is_psd(np.array([[1.0, 2.0], [2.0, 1.0]])):
        fail("the invented table has eigenvalues 3 and -1, so it is not PSD, but is_psd accepted it")
    if sub.is_psd(np.ones((2, 3))):
        fail("a non-square matrix cannot be a Gram matrix, but is_psd accepted it")
    # Both eigenvalues of this matrix are 1, so only the asymmetry should reject it.
    if sub.is_psd(np.array([[1.0, 2.0], [0.0, 1.0]])):
        fail("an asymmetric matrix cannot be a Gram matrix, but is_psd accepted it")


@check("is_psd rejects a polynomial Gram matrix with a negative coef0")
def test_is_psd_rejects_poly(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    K = sub.polynomial_kernel(X, X, 2, -2.0)
    if np.linalg.eigvalsh(K)[0] > -1.0:
        fail("test construction error: expected a clearly negative eigenvalue")
    if sub.is_psd(K):
        fail("the degree-2 polynomial Gram matrix with coef0 = -2 has a clearly negative eigenvalue, but is_psd accepted it")


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


@check("is_psd measures asymmetry with an absolute tolerance, so large entries do not hide a mismatch")
def test_is_psd_absolute_symmetry(sub):
    # Duplicated points with large scores give a rank-one matrix. The upper entry is off by 1e-3,
    # which is far above tol but far below the relative slack np.allclose allows at this size.
    big = np.array([[1e6, 1e6 + 1e-3], [1e6, 1e6]])
    if sub.is_psd(big, tol=1e-8):
        fail("entries differ from their transpose by 1e-3, more than tol=1e-8; use np.allclose(K, K.T, atol=tol, rtol=0.0)")
    tiny_noise = np.array([[1.0, 0.5 + 1e-10], [0.5, 1.0]])
    if not sub.is_psd(tiny_noise, tol=1e-8):
        fail("a mismatch of 1e-10 is below tol=1e-8, so the matrix should count as symmetric")
