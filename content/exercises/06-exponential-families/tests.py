"""Checks for the exponential families and GLM exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def central_diff(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def second_diff(f, x, eps):
    return (f(x + eps) - 2.0 * f(x) + f(x - eps)) / (eps ** 2)


def poisson_dataset(n: int = 400, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))])
    w_true = np.array([0.3, 0.5, -0.4])[:d]
    y = rng.poisson(np.exp(X @ w_true)).astype(float)
    return X, y, w_true


def bernoulli_dataset(n: int = 200, d: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.0, -0.5, 0.3])[:d]
    p = 1.0 / (1.0 + np.exp(-(X @ w_true)))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y, w_true


@check("log_partition reduces to the right formula for each family")
def test_log_partition(sub):
    got = has(sub, "log_partition")
    eta = np.array([-1.0, 0.0, 1.0, 2.0])
    close(got("gaussian", eta), 0.5 * eta ** 2, atol=1e-10, label="gaussian A(eta)")
    close(got("bernoulli", eta), np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0),
          atol=1e-10, label="bernoulli A(eta)")
    close(got("poisson", eta), np.exp(eta), atol=1e-8, label="poisson A(eta)")


@check("mean_from_natural matches a numerical derivative of log_partition")
def test_mean(sub):
    log_partition = has(sub, "log_partition")
    mean_from_natural = has(sub, "mean_from_natural")
    etas = [-1.5, -0.3, 0.0, 0.7, 2.0]
    for family in ("gaussian", "bernoulli", "poisson"):
        for eta0 in etas:
            numeric = central_diff(lambda e: log_partition(family, np.array([e]))[0], eta0, 1e-6)
            analytic = float(mean_from_natural(family, np.array([eta0]))[0])
            if abs(numeric - analytic) > 1e-4 * max(1.0, abs(numeric)):
                fail(
                    f"{family} mean at eta={eta0}: got {analytic:.6g}, numerical "
                    f"derivative of log_partition gives {numeric:.6g}"
                )


@check("variance_from_natural matches a numerical second derivative of log_partition")
def test_variance(sub):
    log_partition = has(sub, "log_partition")
    variance_from_natural = has(sub, "variance_from_natural")
    etas = [-1.5, -0.3, 0.0, 0.7, 2.0]
    for family in ("gaussian", "bernoulli", "poisson"):
        for eta0 in etas:
            numeric = second_diff(lambda e: log_partition(family, np.array([e]))[0], eta0, 1e-3)
            analytic = float(variance_from_natural(family, np.array([eta0]))[0])
            if abs(numeric - analytic) > 5e-3 * max(1.0, abs(numeric)):
                fail(
                    f"{family} variance at eta={eta0}: got {analytic:.6g}, numerical "
                    f"second derivative of log_partition gives {numeric:.6g}"
                )


@check("the Poisson variance function equals its mean function")
def test_poisson_mean_equals_variance(sub):
    mean_from_natural = has(sub, "mean_from_natural")
    variance_from_natural = has(sub, "variance_from_natural")
    eta = np.array([-2.0, -0.5, 0.0, 0.5, 1.5])
    close(
        variance_from_natural("poisson", eta), mean_from_natural("poisson", eta),
        atol=1e-10, label="Poisson: Var = mean",
    )


@check("glm_gradient matches a finite-difference derivative of the negative log-likelihood")
def test_glm_gradient(sub):
    log_partition = has(sub, "log_partition")
    grad = has(sub, "glm_gradient")
    X, y, _ = poisson_dataset(n=60, d=3)
    w = np.array([0.1, 0.2, -0.1])

    def nll(ww):
        eta = X @ ww
        return float(np.sum(log_partition("poisson", eta) - y * eta))

    eps = 1e-6
    numeric = np.zeros_like(w)
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = eps
        numeric[j] = (nll(w + step) - nll(w - step)) / (2 * eps)

    analytic = grad("poisson", w, X, y)
    shape(analytic, (3,), label="glm_gradient output")
    close(analytic, numeric, rtol=1e-4, atol=1e-3, label="Poisson gradient")


@check("glm_fisher_information matches a finite-difference Hessian, and is symmetric")
def test_glm_fisher_information(sub):
    grad = has(sub, "glm_gradient")
    info = has(sub, "glm_fisher_information")
    X, y, _ = poisson_dataset(n=60, d=3)
    w = np.array([0.1, 0.2, -0.1])

    eps = 1e-5
    d = len(w)
    numeric = np.zeros((d, d))
    for j in range(d):
        step = np.zeros(d)
        step[j] = eps
        numeric[:, j] = (
            grad("poisson", w + step, X, y) - grad("poisson", w - step, X, y)
        ) / (2 * eps)

    analytic = info("poisson", w, X)
    shape(analytic, (3, 3), label="glm_fisher_information output")
    close(analytic, analytic.T, atol=1e-8, label="Fisher information should be symmetric")
    close(analytic, numeric, rtol=1e-3, atol=1e-3, label="Poisson Fisher information")


@check("fit_fisher_scoring recovers known coefficients on simulated Poisson counts")
def test_poisson_recovery(sub):
    X, y, w_true = poisson_dataset()
    w_hat, history = has(sub, "fit_fisher_scoring")("poisson", X, y, max_iter=30)
    shape(w_hat, (3,), label="fit_fisher_scoring output")
    if len(history) >= 10:
        fail(f"fisher scoring took {len(history)} iterations, expected fewer than 10")
    close(w_hat, w_true, atol=0.15, label="recovered Poisson coefficients")
    if history != sorted(history, reverse=True):
        fail("the negative log-likelihood should decrease monotonically on this problem")


@check("The Bernoulli case reproduces the logistic Newton step exactly")
def test_bernoulli_matches_logistic_newton(sub):
    X, y, _ = bernoulli_dataset()
    fit = has(sub, "fit_fisher_scoring")

    # At w = 0 every Bernoulli mean is 0.5 and every variance is 0.25, so one
    # step of Fisher scoring is one step of Newton's method on the logistic
    # negative log-likelihood: w1 = -(X^T (0.25 I) X)^{-1} X^T (0.5 - y).
    w1, _ = fit("bernoulli", X, y, max_iter=1)
    gradient_at_zero = X.T @ (0.5 - y)
    hessian_at_zero = 0.25 * (X.T @ X)
    manual_step = -np.linalg.solve(hessian_at_zero, gradient_at_zero)
    close(w1, manual_step, atol=1e-8, label="first Fisher scoring step for Bernoulli")

    # The full run should also converge in single-digit iterations, exactly
    # as Newton's method does on the same problem in the previous lesson.
    w_full, history_full = fit("bernoulli", X, y, max_iter=30)
    if len(history_full) >= 10:
        fail(f"fisher scoring took {len(history_full)} iterations, expected fewer than 10")
    grad = has(sub, "glm_gradient")("bernoulli", w_full, X, y)
    if np.linalg.norm(grad) > 1e-4:
        fail(f"gradient norm at the reported Bernoulli solution is {np.linalg.norm(grad):.3g}")
