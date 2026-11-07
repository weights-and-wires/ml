"""Checks for the kernel ridge and random features exercise."""

import numpy as np

from mlcheck import check, close, fail, shape


def objective(K, y, lam, alpha):
    return float(np.sum((y - K @ alpha) ** 2) + lam * alpha @ K @ alpha)


@check("kernel_ridge_fit solves (K + lam I) alpha = y, including the worked two-point example")
def test_fit_solves_system(sub):
    rng = np.random.default_rng(2)
    n = 15
    A = rng.normal(size=(n, n))
    K = A @ A.T  # positive semidefinite by construction
    y = rng.normal(size=n)
    lam = 0.4
    alpha = sub.kernel_ridge_fit(K, y, lam)
    shape(alpha, (n,), label="alpha")
    close((K + lam * np.eye(n)) @ alpha, y, atol=1e-8, rtol=1e-6, label="(K + lam I) alpha")

    k = np.exp(-0.5)
    K2 = np.array([[1.0, k], [k, 1.0]])
    y2 = np.array([1.0, 2.0])
    alpha2 = sub.kernel_ridge_fit(K2, y2, 1.0)
    close(alpha2, np.array([0.21666095, 0.93429425]), atol=1e-6, label="alpha for lam = 1")
    close(K2 @ alpha2, np.array([0.78333905, 1.06570575]), atol=1e-6, label="training predictions K alpha")


@check("alpha is a stationary point of the objective: its finite-difference gradient is zero")
def test_fit_is_stationary(sub):
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 6, size=8))[:, None]
    K = sub.rbf_kernel(x, x, 1.0)
    y = np.sin(x[:, 0]) + 0.1 * rng.normal(size=8)
    lam = 0.3
    alpha = sub.kernel_ridge_fit(K, y, lam)
    h = 1e-5
    grad = np.zeros(8)
    for i in range(8):
        e = np.zeros(8)
        e[i] = h
        grad[i] = (objective(K, y, lam, alpha + e) - objective(K, y, lam, alpha - e)) / (2 * h)
    close(grad, np.zeros(8), atol=1e-5, label="finite-difference gradient of the objective at alpha")
    best = objective(K, y, lam, alpha)
    for _ in range(5):
        if objective(K, y, lam, alpha + 0.05 * rng.normal(size=8)) < best:
            fail("a random nudge of alpha lowered the objective, so alpha is not the minimiser")


@check("Training predictions equal y - lam alpha; tiny lam interpolates and huge lam shrinks")
def test_lambda_behaviour(sub):
    x = np.array([0.51, 2.37, 3.21, 3.82, 5.93, 6.13])[:, None]
    y = np.array([0.36, 0.73, -0.39, -0.58, -0.30, 0.17])
    K = sub.rbf_kernel(x, x, 1.0)
    lam = 0.1
    alpha = sub.kernel_ridge_fit(K, y, lam)
    close(K @ alpha, y - lam * alpha, atol=1e-8, label="K alpha against y - lam alpha")
    tiny = sub.kernel_ridge_fit(K, y, 1e-8)
    close(K @ tiny, y, atol=1e-5, label="training predictions at lam = 1e-8")
    huge = sub.kernel_ridge_fit(K, y, 1e3)
    if np.linalg.norm(K @ huge) > 0.05 * np.linalg.norm(y):
        fail(f"with lam = 1000 the predictions should shrink toward 0, but their norm is {np.linalg.norm(K @ huge):.3f} against {np.linalg.norm(y):.3f} for y")


@check("kernel_ridge_predict matches a hand-written sum of alpha_i k(x, x_i)")
def test_predict(sub):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 2))
    alpha = rng.normal(size=5)
    queries = rng.normal(size=(3, 2))
    K_test = sub.rbf_kernel(queries, X, 0.8)
    got = sub.kernel_ridge_predict(K_test, alpha)
    shape(got, (3,), label="predictions")
    want = np.array([sum(alpha[i] * np.exp(-0.8 * np.sum((q - X[i]) ** 2)) for i in range(5)) for q in queries])
    close(got, want, atol=1e-10, label="sum of weighted similarities")


@check("A linear kernel reproduces primal ridge regression through w = X^T alpha")
def test_matches_primal_ridge(sub):
    # Hand case: x = (1, 2), y = (1, 2), lam = 1 gives alpha = (1/6, 1/3) and w = 5/6.
    X1 = np.array([[1.0], [2.0]])
    a1 = sub.kernel_ridge_fit(X1 @ X1.T, np.array([1.0, 2.0]), 1.0)
    close(a1, np.array([1 / 6, 1 / 3]), atol=1e-10, label="alpha for the one-dimensional example")
    close(sub.implied_weights(X1, a1), np.array([5 / 6]), atol=1e-10, label="w for the one-dimensional example")

    rng = np.random.default_rng(5)
    n, d = 20, 5
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    lam = 0.7
    alpha = sub.kernel_ridge_fit(X @ X.T, y, lam)
    w = sub.implied_weights(X, alpha)
    shape(w, (d,), label="implied weights")
    w_primal = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    close(w, w_primal, atol=1e-8, rtol=1e-6, label="X^T alpha against primal ridge")
    X_test = rng.normal(size=(8, d))
    close(sub.kernel_ridge_predict(X_test @ X.T, alpha), X_test @ w_primal, atol=1e-8, rtol=1e-6, label="predictions")


@check("representer_check reproduces the hand example (J is 2 and 3) and sums squared errors over training rows only")
def test_representer_formula(sub):
    K = np.eye(2)
    y = np.array([2.0])
    close(sub.representer_check(K, y, 1.0, np.array([1.0, 0.0])), 2.0, atol=1e-12, label="J for c = (1, 0)")
    close(sub.representer_check(K, y, 1.0, np.array([1.0, 1.0])), 3.0, atol=1e-12, label="J for c = (1, 1)")

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


@check("random_fourier_features returns shape (n, D) bounded by sqrt(2 / D), and two same-seed calls share one feature map")
def test_rff_shape_and_shared_draw(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 3))
    Z = sub.random_fourier_features(X, 50, 0.5, np.random.default_rng(0))
    shape(Z, (9, 50), label="random Fourier features")
    bound = np.sqrt(2.0 / 50)
    if np.abs(Z).max() > bound + 1e-12:
        fail(f"every entry is sqrt(2/D) times a cosine, so |Z| <= {bound:.4f}, but the largest entry is {np.abs(Z).max():.4f}")

    X12 = rng.normal(size=(12, 3))
    Z_train = sub.random_fourier_features(X12, 30, 0.7, np.random.default_rng(99))
    Z_test = sub.random_fourier_features(X12[:4], 30, 0.7, np.random.default_rng(99))
    close(Z_test, Z_train[:4], atol=1e-12, label="features of the same rows from two same-seed calls")


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


@check("ridge_on_features has zero gradient at its answer, and its predictions approach exact kernel ridge as D grows")
def test_ridge_on_features(sub):
    rng = np.random.default_rng(3)
    n, D = 25, 6
    Z = rng.normal(size=(n, D))
    y = rng.normal(size=n)
    lam = 0.5

    def objective_w(w):
        return float(np.sum((y - Z @ w) ** 2) + lam * w @ w)

    w = sub.ridge_on_features(Z, y, lam)
    shape(w, (D,), label="weights")
    h = 1e-6
    grad = np.array([(objective_w(w + h * e) - objective_w(w - h * e)) / (2 * h) for e in np.eye(D)])
    close(grad, np.zeros(D), atol=1e-5, label="finite-difference gradient at w")
    close(w, np.linalg.solve(Z.T @ Z + lam * np.eye(D), Z.T @ y), atol=1e-8, label="normal-equation solution")

    rng = np.random.default_rng(6)
    m = 40
    x = np.sort(rng.uniform(0, 6, size=m))[:, None]
    ys = np.sin(x[:, 0]) + 0.1 * rng.normal(size=m)
    x_test = np.linspace(0, 6, 25)[:, None]
    gamma, lam = 1.0, 0.1
    alpha = sub.kernel_ridge_fit(sub.rbf_kernel(x, x, gamma), ys, lam)
    exact = sub.rbf_kernel(x_test, x, gamma) @ alpha

    gaps = []
    for feats in (50, 3000):
        Zt = sub.random_fourier_features(x, feats, gamma, np.random.default_rng(7))
        Z_test = sub.random_fourier_features(x_test, feats, gamma, np.random.default_rng(7))
        wt = sub.ridge_on_features(Zt, ys, lam)
        gaps.append(np.abs(Z_test @ wt - exact).max())
    if gaps[1] >= gaps[0]:
        fail(f"the gap to exact kernel ridge should shrink as D grows, got {gaps[0]:.4f} at D=50 and {gaps[1]:.4f} at D=3000")
    if gaps[1] > 0.03:
        fail(f"at D=3000 predictions should be within 0.03 of exact kernel ridge, but the largest gap is {gaps[1]:.4f}")
