"""Checks for the logistic regression exercise."""

import numpy as np

from mlcheck import check, close, fail, finite, has, shape


def separable_dataset(n: int = 150, d: int = 3, seed: int = 5):
    """A design matrix and labels that a linear boundary separates exactly."""
    rng = np.random.default_rng(seed)
    w_sep = np.array([2.0, -1.5, 1.0])[:d]
    X = rng.normal(size=(n, d))
    margin = X @ w_sep
    # Redraw the rare point too close to the boundary, so no seed makes this
    # flaky and the margin stays bounded away from zero.
    while np.abs(margin).min() < 0.05:
        X = rng.normal(size=(n, d))
        margin = X @ w_sep
    y = (margin > 0).astype(float)
    return X, y


def logistic_dataset(n: int = 300, d: int = 4, seed: int = 0):
    """A well-conditioned, non-separable logistic regression problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5, 0.8])[:d]

    def sigmoid_ref(z):
        return 1.0 / (1.0 + np.exp(-z))

    y = (rng.uniform(size=n) < sigmoid_ref(X @ w_true)).astype(float)
    return X, y, w_true


def numerical_hessian(grad_func, w, eps=1e-5):
    """Central-difference Jacobian of a vector-valued gradient function."""
    d = w.shape[0]
    H = np.zeros((d, d))
    for j in range(d):
        step = np.zeros(d)
        step[j] = eps
        H[:, j] = (grad_func(w + step) - grad_func(w - step)) / (2 * eps)
    return H


@check("sigmoid is finite and correct at magnitudes as large as 800")
def test_sigmoid_stability(sub):
    got = has(sub, "sigmoid")
    z = np.array([-800.0, -20.0, 0.0, 20.0, 800.0])
    out = got(z)
    finite(out, label="sigmoid")
    close(out, np.array([0.0, 2.061153622438558e-09, 0.5, 0.9999999979388463, 1.0]), atol=1e-9)


@check("log_loss is finite at extreme logits and correct at z = 0")
def test_log_loss(sub):
    got = has(sub, "log_loss")
    X_extreme = np.array([[1.0], [1.0]])
    y_extreme = np.array([1.0, 0.0])
    finite(got(np.array([800.0]), X_extreme, y_extreme), label="log_loss at z = 800")
    finite(got(np.array([-800.0]), X_extreme, y_extreme), label="log_loss at z = -800")

    # At w = 0 every logit is 0, so each term is log(2), independent of y.
    X_flat = np.ones((10, 2))
    y_flat = np.zeros(10)
    value = got(np.zeros(2), X_flat, y_flat)
    close(value, 10 * np.log(2.0), atol=1e-8, label="loss at w = 0")

    # The ridge term adds (ridge / 2) ||w||^2 exactly.
    w = np.array([3.0, -4.0])
    with_ridge = got(w, X_flat, y_flat, ridge=2.0)
    without_ridge = got(w, X_flat, y_flat, ridge=0.0)
    close(with_ridge - without_ridge, 1.0 * (9.0 + 16.0), atol=1e-6, label="ridge penalty")


@check("gradient matches a finite-difference derivative of log_loss")
def test_gradient(sub):
    from mlcheck import finite_difference

    log_loss = has(sub, "log_loss")
    grad = has(sub, "gradient")
    X, y, _ = logistic_dataset(n=60, d=4)
    w = np.array([0.3, -0.2, 0.1, 0.5])

    for ridge in (0.0, 0.5):
        numeric = finite_difference(lambda ww: log_loss(ww, X, y, ridge), w)
        analytic = grad(w, X, y, ridge)
        shape(analytic, (4,), label=f"gradient shape, ridge={ridge}")
        close(analytic, numeric, rtol=1e-5, atol=1e-5, label=f"gradient, ridge={ridge}")


@check("hessian matches a finite-difference Jacobian of gradient")
def test_hessian_finite_difference(sub):
    grad = has(sub, "gradient")
    hess = has(sub, "hessian")
    X, y, _ = logistic_dataset(n=60, d=4)
    w = np.array([0.3, -0.2, 0.1, 0.5])

    for ridge in (0.0, 0.5):
        numeric = numerical_hessian(lambda ww: grad(ww, X, y, ridge), w)
        analytic = hess(w, X, y, ridge)
        shape(analytic, (4, 4), label=f"hessian shape, ridge={ridge}")
        close(analytic, numeric, rtol=1e-4, atol=1e-4, label=f"hessian, ridge={ridge}")


@check("hessian is positive semidefinite")
def test_hessian_psd(sub):
    hess = has(sub, "hessian")
    X, y, _ = logistic_dataset(n=80, d=4, seed=2)
    rng = np.random.default_rng(3)
    w = rng.normal(size=4)
    H = hess(w, X, y, ridge=0.0)
    close(H, H.T, atol=1e-10, label="hessian should be symmetric")
    eigenvalues = np.linalg.eigvalsh(H)
    smallest = float(eigenvalues.min())
    if smallest < -1e-6:
        fail(f"hessian has a negative eigenvalue {smallest:.3g}, so it is not PSD")


@check("Newton converges in under 10 iterations where gradient descent needs hundreds")
def test_newton_vs_gradient_descent(sub):
    X, y, _ = logistic_dataset()
    w_newton, history_newton = has(sub, "fit_newton")(X, y, max_iter=30)
    w_gd, history_gd = has(sub, "fit_gradient_descent")(X, y, lr=0.005, max_iter=5000, tol=1e-9)

    if len(history_newton) >= 10:
        fail(f"fit_newton took {len(history_newton)} iterations, expected fewer than 10")
    if len(history_gd) < 150:
        fail(
            f"fit_gradient_descent took only {len(history_gd)} iterations; the point of "
            "the comparison is that it needs hundreds where Newton needs single digits"
        )
    close(
        w_newton, w_gd, atol=1e-3,
        label="both routes should reach the same optimum",
    )
    close(
        history_newton[-1], history_gd[-1], atol=1e-4,
        label="both routes should reach the same loss",
    )


@check("fit_newton reaches a first-order stationary point")
def test_newton_optimality(sub):
    X, y, _ = logistic_dataset(n=200, d=3, seed=9)
    grad = has(sub, "gradient")
    w, history = has(sub, "fit_newton")(X, y, max_iter=30, ridge=1e-6)
    g = grad(w, X, y, ridge=1e-6)
    residual = float(np.linalg.norm(g))
    if residual > 1e-4:
        fail(f"gradient norm at the reported solution is {residual:.3g}, expected near 0")
    if history != sorted(history, reverse=True):
        fail("the loss history should decrease monotonically for a converging run")


@check("Separable data sends the unregularised coefficient norm up while ridge holds it back")
def test_separability(sub):
    X, y = separable_dataset()
    fit = has(sub, "fit_gradient_descent")

    w_early_plain, _ = fit(X, y, lr=0.05, max_iter=300, ridge=0.0, tol=0.0)
    w_late_plain, history_plain = fit(X, y, lr=0.05, max_iter=3000, ridge=0.0, tol=0.0)
    w_late_ridge, history_ridge = fit(X, y, lr=0.05, max_iter=3000, ridge=1.0, tol=0.0)

    norm_early = float(np.linalg.norm(w_early_plain))
    norm_late = float(np.linalg.norm(w_late_plain))
    norm_ridge = float(np.linalg.norm(w_late_ridge))

    if norm_late < 1.3 * norm_early:
        fail(
            "the unregularised coefficient norm should keep growing as training "
            f"continues on separable data: it went from {norm_early:.3g} at 300 "
            f"iterations to only {norm_late:.3g} at 3000"
        )
    if norm_ridge > 0.5 * norm_late:
        fail(
            f"the ridge-regularised norm ({norm_ridge:.3g}) should stay far smaller "
            f"than the unregularised one ({norm_late:.3g}) on separable data"
        )
    if history_plain[-1] > history_plain[0] * 0.1:
        fail(
            "the unregularised loss should fall a long way toward zero on separable "
            f"data, from {history_plain[0]:.3g} to {history_plain[-1]:.3g}"
        )
    if history_ridge[-1] < history_plain[-1]:
        fail(
            "the ridge-regularised loss should settle above the unregularised loss, "
            "because the penalty stops it from driving the negative log-likelihood to zero"
        )
