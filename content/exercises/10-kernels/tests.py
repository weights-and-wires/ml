"""Checks for the kernels and representer theorem exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


@check("rbf_kernel matches a direct computation on a hand-built example")
def test_rbf_direct(sub):
    got = has(sub, "rbf_kernel")
    X1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    X2 = np.array([[0.0, 1.0]])
    gamma = 0.5
    # squared distances: (0,0)-(0,1) -> 1, (1,0)-(0,1) -> 2
    want = np.exp(-gamma * np.array([[1.0], [2.0]]))
    close(got(X1, X2, gamma), want, atol=1e-10, label="RBF kernel values")


@check("polynomial_kernel matches a direct computation on a hand-built example")
def test_polynomial_direct(sub):
    got = has(sub, "polynomial_kernel")
    X1 = np.array([[1.0, 2.0], [0.0, 1.0]])
    X2 = np.array([[3.0, 0.0]])
    degree, coef0 = 3, 1.0
    # dot products: 1*3+2*0=3, 0*3+1*0=0
    want = np.array([[(3.0 + 1.0) ** 3], [(0.0 + 1.0) ** 3]])
    close(got(X1, X2, degree, coef0), want, atol=1e-10, label="polynomial kernel values")


@check("rbf_kernel and polynomial_kernel produce PSD Gram matrices on random inputs")
def test_kernels_are_psd(sub):
    rbf = has(sub, "rbf_kernel")
    poly = has(sub, "polynomial_kernel")
    psd = has(sub, "is_psd")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 5))

    K_rbf = rbf(X, X, 0.3)
    shape(K_rbf, (25, 25), label="RBF Gram matrix")
    if not psd(K_rbf):
        fail("the RBF kernel's Gram matrix should be PSD on any input, but is_psd rejected it")

    K_poly = poly(X, X, 3, 1.0)
    if not psd(K_poly):
        fail("this polynomial kernel's Gram matrix should be PSD on any input, but is_psd rejected it")


@check("is_psd rejects a matrix that is not positive semidefinite")
def test_is_psd_rejects(sub):
    got = has(sub, "is_psd")
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    indefinite = A + A.T  # symmetric, but generically has both signs of eigenvalue
    eigenvalues = np.linalg.eigvalsh(indefinite)
    if eigenvalues.min() > -0.1:
        # Extremely unlikely with this seed, but guard against a degenerate draw.
        indefinite = indefinite - (eigenvalues.min() + 1.0) * np.eye(6)
    if got(indefinite):
        fail("a symmetric matrix with a clearly negative eigenvalue was accepted as PSD")

    asymmetric = np.array([[1.0, 2.0], [0.0, 1.0]])
    if got(asymmetric):
        fail("an asymmetric matrix was accepted as PSD")


@check("kernel_ridge_fit solves (K + lam I) alpha = y")
def test_kernel_ridge_fit_solves_system(sub):
    got = has(sub, "kernel_ridge_fit")
    rng = np.random.default_rng(2)
    n = 15
    A = rng.normal(size=(n, n))
    K = A @ A.T  # PSD by construction
    y = rng.normal(size=n)
    lam = 0.4

    alpha = got(K, y, lam)
    shape(alpha, (n,), label="alpha")
    residual = (K + lam * np.eye(n)) @ alpha
    close(residual, y, atol=1e-8, rtol=1e-6, label="(K + lam I) alpha")


@check("Kernel ridge regression with a linear kernel matches primal ridge regression exactly")
def test_matches_primal_ridge(sub):
    fit = has(sub, "kernel_ridge_fit")
    predict = has(sub, "kernel_ridge_predict")
    rng = np.random.default_rng(3)
    n, d = 20, 5
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    lam = 0.7

    K = X @ X.T
    alpha = fit(K, y, lam)
    w_from_kernel = X.T @ alpha
    w_primal = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    close(w_from_kernel, w_primal, atol=1e-6, rtol=1e-6, label="implied weight vector w = X^T alpha")

    X_test = rng.normal(size=(8, d))
    K_test = X_test @ X.T
    close(
        predict(K_test, alpha),
        X_test @ w_primal,
        atol=1e-6,
        rtol=1e-6,
        label="predictions against the primal solution",
    )


@check("random_fourier_features approximate the RBF Gram matrix, with error shrinking as D grows")
def test_random_fourier_features(sub):
    rbf = has(sub, "rbf_kernel")
    got = has(sub, "random_fourier_features")
    rng = np.random.default_rng(4)
    n, d = 30, 4
    X = rng.normal(size=(n, d))
    gamma = 0.4
    K = rbf(X, X, gamma)

    Z_small = got(X, 40, gamma, np.random.default_rng(10))
    shape(Z_small, (n, 40), label="random Fourier features, D=40")
    error_small = np.abs(K - Z_small @ Z_small.T).mean()

    Z_large = got(X, 4000, gamma, np.random.default_rng(11))
    error_large = np.abs(K - Z_large @ Z_large.T).mean()

    if error_large >= error_small:
        fail(
            f"mean absolute Gram matrix error should shrink as D grows: got {error_small:.4f} "
            f"at D=40 and {error_large:.4f} at D=4000"
        )
    if error_large > 0.05:
        fail(f"at D=4000 the mean absolute error against the exact RBF kernel should be small, got {error_large:.4f}")


@check("representer_check shows an out-of-span component strictly and exactly increases the objective")
def test_representer_theorem(sub):
    rbf = has(sub, "rbf_kernel")
    fit = has(sub, "kernel_ridge_fit")
    got = has(sub, "representer_check")

    rng = np.random.default_rng(5)
    n, m, d = 6, 10, 2
    X_train = rng.normal(size=(n, d))
    X_aux = 1.5 * rng.normal(size=(m, d))
    y = rng.normal(size=n)
    lam, gamma = 0.3, 0.7

    K_tt = rbf(X_train, X_train, gamma)
    K_ta = rbf(X_train, X_aux, gamma)
    K_aa = rbf(X_aux, X_aux, gamma)
    K_big = np.block([[K_tt, K_ta], [K_ta.T, K_aa]])

    alpha = fit(K_tt, y, lam)
    alpha_full = np.concatenate([alpha, np.zeros(m)])
    baseline = got(K_big, y, lam, alpha_full)

    # A direction, supported only on the auxiliary points, that makes the
    # resulting function vanish at every training point: it lies entirely
    # outside the span of the training kernel sections.
    _, _, vt = np.linalg.svd(K_ta)
    direction = vt[-1]
    if np.abs(K_ta @ direction).max() > 1e-6:
        fail("test construction error: the auxiliary direction does not vanish at the training points")

    for t in (0.5, 1.0, -1.0, 2.0):
        coeffs = alpha_full + t * np.concatenate([np.zeros(n), direction])
        perturbed = got(K_big, y, lam, coeffs)
        if perturbed <= baseline + 1e-9:
            fail(
                f"adding an out-of-span component with t={t} should strictly increase the "
                f"regularised objective ({baseline:.6f}), but it gave {perturbed:.6f}"
            )
        predicted_gap = lam * t**2 * float(direction @ K_aa @ direction)
        close(
            perturbed - baseline,
            predicted_gap,
            atol=1e-6,
            rtol=1e-5,
            label=f"objective gap at t={t}",
        )
