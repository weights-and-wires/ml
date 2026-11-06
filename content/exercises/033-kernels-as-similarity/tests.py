"""Checks for the kernels-as-similarity exercise."""

import numpy as np

from mlcheck import check, close, fail, shape


@check("rbf_kernel matches a direct computation on a hand-built example")
def test_rbf_direct(sub):
    X1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    X2 = np.array([[0.0, 1.0]])
    gamma = 0.5
    # Squared distances: (0,0)-(0,1) is 1, and (1,0)-(0,1) is 2.
    want = np.exp(-gamma * np.array([[1.0], [2.0]]))
    got = sub.rbf_kernel(X1, X2, gamma)
    shape(got, (2, 1), label="rbf_kernel output")
    close(got, want, atol=1e-10, label="RBF kernel values")


@check("rbf_kernel of a set with itself is symmetric, has a unit diagonal, and never exceeds 1")
def test_rbf_gram_properties(sub):
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


@check("polynomial_kernel matches a hand-worked example: (1,2) and (3,1) give 36 at degree 2")
def test_polynomial_direct(sub):
    X1 = np.array([[1.0, 2.0], [0.0, 1.0]])
    X2 = np.array([[3.0, 1.0]])
    got = sub.polynomial_kernel(X1, X2, 2, 1.0)
    shape(got, (2, 1), label="polynomial_kernel output")
    # Dot products: 1*3 + 2*1 = 5 and 0*3 + 1*1 = 1.
    close(got, np.array([[36.0], [4.0]]), atol=1e-10, label="degree-2 polynomial kernel values")
    cubic = sub.polynomial_kernel(X1, X2, 3, 2.0)
    close(cubic, np.array([[7.0**3], [3.0**3]]), atol=1e-10, label="degree-3 polynomial kernel with coef0 2")


@check("polynomial_kernel with degree 1 and coef0 0 is the plain dot-product (linear) kernel")
def test_linear_kernel(sub):
    rng = np.random.default_rng(2)
    X1 = rng.normal(size=(6, 4))
    X2 = rng.normal(size=(9, 4))
    got = sub.polynomial_kernel(X1, X2, 1, 0.0)
    shape(got, (6, 9), label="cross-kernel matrix")
    close(got, X1 @ X2.T, atol=1e-10, label="linear kernel")


@check("poly_features_degree2 returns the six documented columns")
def test_features_shape_and_values(sub):
    X = np.array([[1.0, 2.0], [3.0, 1.0]])
    phi = sub.poly_features_degree2(X)
    shape(phi, (2, 6), label="feature matrix")
    r = np.sqrt(2.0)
    want = np.array([[1.0, r * 1.0, r * 2.0, 1.0, r * 2.0, 4.0], [1.0, r * 3.0, r * 1.0, 9.0, r * 3.0, 1.0]])
    close(phi, want, atol=1e-10, label="features of (1,2) and (3,1)")


@check("The kernel trick: explicit features and the polynomial kernel give the same Gram matrix")
def test_kernel_trick(sub):
    rng = np.random.default_rng(3)
    X1 = rng.normal(size=(7, 2))
    X2 = rng.normal(size=(5, 2))
    explicit = sub.poly_features_degree2(X1) @ sub.poly_features_degree2(X2).T
    shortcut = sub.polynomial_kernel(X1, X2, 2, 1.0)
    shape(explicit, (7, 5), label="explicit-feature Gram matrix")
    close(explicit, shortcut, atol=1e-10, rtol=1e-10, label="phi(X1) phi(X2)^T against (X1 X2^T + 1)^2")
