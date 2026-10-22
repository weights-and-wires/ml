"""Checks for the representer theorem and random features exercise."""

import numpy as np

from mlcheck import check, close, fail, shape


@check("representer_check reproduces the hand example: J is 2 for c = (1, 0) and 3 for c = (1, 1)")
def test_representer_hand(sub):
    K = np.eye(2)
    y = np.array([2.0])
    close(sub.representer_check(K, y, 1.0, np.array([1.0, 0.0])), 2.0, atol=1e-12, label="J for c = (1, 0)")
    close(sub.representer_check(K, y, 1.0, np.array([1.0, 1.0])), 3.0, atol=1e-12, label="J for c = (1, 1)")


@check("representer_check sums squared errors over training rows only and adds lam times c^T K c")
def test_representer_formula(sub):
    rng = np.random.default_rng(0)
    N, n = 7, 4
    A = rng.normal(size=(N, N))
    K = A @ A.T
    y = rng.normal(size=n)
    c = rng.normal(size=N)
    lam = 0.35
    pred = K @ c
    want = sum((y[i] - pred[i]) ** 2 for i in range(n)) + lam * sum(c[i] * K[i, j] * c[j] for i in range(N) for j in range(N))
    close(sub.representer_check(K, y, lam, c), want, atol=1e-8, rtol=1e-8, label="objective")


@check("An extra component that vanishes at the training points raises the objective by exactly lam t^2 d^T K_aa d")
def test_representer_theorem(sub):
    rng = np.random.default_rng(5)
    n, m, d = 6, 10, 2
    X_train = rng.normal(size=(n, d))
    X_aux = 1.5 * rng.normal(size=(m, d))
    y = rng.normal(size=n)
    lam, gamma = 0.3, 0.7

    K_tt = sub.rbf_kernel(X_train, X_train, gamma)
    K_ta = sub.rbf_kernel(X_train, X_aux, gamma)
    K_aa = sub.rbf_kernel(X_aux, X_aux, gamma)
    K_big = np.block([[K_tt, K_ta], [K_ta.T, K_aa]])

    alpha = sub.kernel_ridge_fit(K_tt, y, lam)
    alpha_full = np.concatenate([alpha, np.zeros(m)])
    baseline = sub.representer_check(K_big, y, lam, alpha_full)

    # A direction on the extra centres whose function is zero at every training point.
    _, _, vt = np.linalg.svd(K_ta)
    direction = vt[-1]
    if np.abs(K_ta @ direction).max() > 1e-6:
        fail("test construction error: the extra direction does not vanish at the training points")

    for t in (0.5, 1.0, -1.0, 2.0):
        coeffs = alpha_full + t * np.concatenate([np.zeros(n), direction])
        perturbed = sub.representer_check(K_big, y, lam, coeffs)
        if perturbed <= baseline + 1e-9:
            fail(f"adding an extra component with t={t} should strictly raise the objective ({baseline:.6f}), but it gave {perturbed:.6f}")
        predicted_gap = lam * t**2 * float(direction @ K_aa @ direction)
        close(perturbed - baseline, predicted_gap, atol=1e-6, rtol=1e-5, label=f"objective gap at t={t}")


@check("random_fourier_features returns shape (n, D) with entries bounded by sqrt(2 / D)")
def test_rff_shape(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 3))
    Z = sub.random_fourier_features(X, 50, 0.5, np.random.default_rng(0))
    shape(Z, (9, 50), label="random Fourier features")
    bound = np.sqrt(2.0 / 50)
    if np.abs(Z).max() > bound + 1e-12:
        fail(f"every entry is sqrt(2/D) times a cosine, so |Z| <= {bound:.4f}, but the largest entry is {np.abs(Z).max():.4f}")


@check("The random-feature Gram matrix approaches the exact RBF Gram matrix at about a 1/sqrt(D) rate")
def test_rff_approximation(sub):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 4))
    gamma = 0.4
    K = sub.rbf_kernel(X, X, gamma)

    def error(D, seed):
        Z = sub.random_fourier_features(X, D, gamma, np.random.default_rng(seed))
        return np.abs(K - Z @ Z.T).mean()

    e40, e4000 = error(40, 10), error(4000, 11)
    if e4000 >= e40:
        fail(f"mean absolute Gram matrix error should shrink as D grows: got {e40:.4f} at D=40 and {e4000:.4f} at D=4000")
    if e4000 > 0.05:
        fail(f"at D=4000 the mean absolute error should be small, got {e4000:.4f}")
    e100, e1600 = error(100, 12), error(1600, 13)
    ratio = e100 / e1600
    if not 2.0 < ratio < 8.0:
        fail(f"sixteen times as many features should cut the error by about 4, but the ratio was {ratio:.2f} ({e100:.4f} to {e1600:.4f})")


@check("Two calls with the same seed use the same random draw, so train and test features share a map")
def test_rff_shared_draw(sub):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))
    X_test = X[:4]
    Z_train = sub.random_fourier_features(X, 30, 0.7, np.random.default_rng(99))
    Z_test = sub.random_fourier_features(X_test, 30, 0.7, np.random.default_rng(99))
    close(Z_test, Z_train[:4], atol=1e-12, label="features of the same rows from two same-seed calls")


@check("ridge_on_features has zero gradient at its answer, checked with finite differences")
def test_ridge_on_features(sub):
    rng = np.random.default_rng(3)
    n, D = 25, 6
    Z = rng.normal(size=(n, D))
    y = rng.normal(size=n)
    lam = 0.5

    def objective(w):
        return float(np.sum((y - Z @ w) ** 2) + lam * w @ w)

    w = sub.ridge_on_features(Z, y, lam)
    shape(w, (D,), label="weights")
    h = 1e-6
    grad = np.array([(objective(w + h * e) - objective(w - h * e)) / (2 * h) for e in np.eye(D)])
    close(grad, np.zeros(D), atol=1e-5, label="finite-difference gradient at w")
    close(w, np.linalg.solve(Z.T @ Z + lam * np.eye(D), Z.T @ y), atol=1e-8, label="normal-equation solution")


@check("Ridge on random features approaches exact kernel ridge predictions as D grows")
def test_rff_ridge_matches_kernel_ridge(sub):
    rng = np.random.default_rng(6)
    n = 40
    x = np.sort(rng.uniform(0, 6, size=n))[:, None]
    y = np.sin(x[:, 0]) + 0.1 * rng.normal(size=n)
    x_test = np.linspace(0, 6, 25)[:, None]
    gamma, lam = 1.0, 0.1
    alpha = sub.kernel_ridge_fit(sub.rbf_kernel(x, x, gamma), y, lam)
    exact = sub.rbf_kernel(x_test, x, gamma) @ alpha

    gaps = []
    for D in (50, 3000):
        Z = sub.random_fourier_features(x, D, gamma, np.random.default_rng(7))
        Z_test = sub.random_fourier_features(x_test, D, gamma, np.random.default_rng(7))
        w = sub.ridge_on_features(Z, y, lam)
        gaps.append(np.abs(Z_test @ w - exact).max())
    if gaps[1] >= gaps[0]:
        fail(f"the gap to exact kernel ridge should shrink as D grows, got {gaps[0]:.4f} at D=50 and {gaps[1]:.4f} at D=3000")
    if gaps[1] > 0.03:
        fail(f"at D=3000 predictions should be within 0.03 of exact kernel ridge, but the largest gap is {gaps[1]:.4f}")
