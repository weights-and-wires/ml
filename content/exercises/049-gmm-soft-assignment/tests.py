"""Checks for the Gaussian mixture E-step exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def two_points():
    X = np.array([[0.0], [2.0]])
    means = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])
    return X, means, covs, weights


def spd(rng, d):
    A = rng.normal(size=(d, d))
    return A @ A.T + 0.5 * np.eye(d)


@check("log_gaussian gives -0.918939 at the mean of a standard normal, and -2.918939 two away")
def test_log_gaussian_hand(sub):
    f = has(sub, "log_gaussian")
    got = np.asarray(f(np.array([[0.0], [2.0]]), np.array([0.0]), np.array([[1.0]])))
    shape(got, (2,), label="log densities")
    close(got, np.array([-0.5 * np.log(2 * np.pi), -0.5 * np.log(2 * np.pi) - 2.0]), atol=1e-12,
          label="log densities at 0 and 2")


@check("log_gaussian matches a direct formula for a tilted 2-D covariance")
def test_log_gaussian_tilted(sub):
    f = has(sub, "log_gaussian")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 2)) * 2
    mean = np.array([0.5, -1.0])
    cov = np.array([[2.0, 1.2], [1.2, 1.0]])
    want = np.empty(25)
    inv = np.linalg.inv(cov)
    for i in range(25):
        g = X[i] - mean
        want[i] = -0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(cov)) + g @ inv @ g)
    close(np.asarray(f(X, mean, cov)), want, atol=1e-10, label="log density")


@check("log_gaussian in 3 dimensions equals a sum of 1-D log densities when the covariance is diagonal")
def test_log_gaussian_diagonal(sub):
    f = has(sub, "log_gaussian")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    mean = np.array([0.1, 0.2, -0.3])
    var = np.array([0.5, 2.0, 1.5])
    want = np.sum(-0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var), axis=1)
    close(np.asarray(f(X, mean, np.diag(var))), want, atol=1e-10, label="log density")


@check("logsumexp_rows survives scores near -1000 and matches the naive formula on small ones")
def test_logsumexp(sub):
    f = has(sub, "logsumexp_rows")
    s = np.array([[-1000.0, -1001.0], [0.5, -0.5], [-800.0, -800.0]])
    got = np.asarray(f(s))
    shape(got, (3,), label="row totals")
    # A square input catches a row maximum with shape (n,) instead of (n, 1):
    # it broadcasts along the wrong axis and nothing complains.
    square = np.array([[-1000.0, -1001.0], [-5.0, -6.0]])
    got_square = np.asarray(f(square))
    want_square = np.array([-1000.0 + np.log1p(np.exp(-1.0)), -5.0 + np.log1p(np.exp(-1.0))])
    close(got_square, want_square, atol=1e-10, label="row totals when n equals k")
    if not np.all(np.isfinite(got)):
        fail(f"a row total is not finite: {got}. The scores -1000 exponentiate to 0, so subtract the row maximum first.")
    want = np.array([-1000.0 + np.log1p(np.exp(-1.0)), np.log(np.exp(0.5) + np.exp(-0.5)), -800.0 + np.log(2.0)])
    close(got, want, atol=1e-10, label="log of row sums of exp")
    before = s.copy()
    f(s)
    close(s, before, atol=0.0, label="input array after the call")


@check("gmm_e_step reproduces the chapter: shares 0.880797 and 0.119203, log L = -2.970")
def test_e_step_hand(sub):
    e_step = has(sub, "gmm_e_step")
    X, means, covs, weights = two_points()
    resp, ll = e_step(X, means, covs, weights)
    shape(resp, (2, 2), label="responsibilities")
    close(resp, np.array([[0.880797, 0.119203], [0.119203, 0.880797]]), atol=1e-6, label="responsibilities")
    if not isinstance(ll, float):
        fail(f"expected a Python float for the log-likelihood, got {type(ll).__name__}")
    almost(ll, -2.970315, tol=1e-6, label="log-likelihood")


@check("gmm_e_step responsibilities follow Bayes' rule with unequal weights and a tilted blob")
def test_e_step_bayes(sub):
    e_step = has(sub, "gmm_e_step")
    log_gaussian = has(sub, "log_gaussian")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 2)) * 2
    means = np.array([[0.0, 0.0], [3.0, 1.0], [-2.0, 2.0]])
    covs = np.stack([np.eye(2), np.array([[2.0, 1.2], [1.2, 1.0]]), 0.5 * np.eye(2)])
    weights = np.array([0.2, 0.5, 0.3])
    resp, ll = e_step(X, means, covs, weights)
    dens = np.stack([weights[j] * np.exp(log_gaussian(X, means[j], covs[j])) for j in range(3)], axis=1)
    close(resp, dens / dens.sum(axis=1, keepdims=True), atol=1e-9, label="weight times density, normalised")
    close(resp.sum(axis=1), np.ones(12), atol=1e-12, label="row sums")
    almost(ll, float(np.sum(np.log(dens.sum(axis=1)))), tol=1e-8, label="log-likelihood")


@check("gmm_e_step stays finite when a point is hundreds of units from every blob")
def test_e_step_far(sub):
    e_step = has(sub, "gmm_e_step")
    # The middle point is 500 away from both blobs, so both scores are about
    # -12 million and a naive exp() of either one is exactly 0.
    X = np.array([[0.1], [500.0], [999.9]])
    means = np.array([[0.0], [1000.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])
    resp, ll = e_step(X, means, covs, weights)
    if not (np.all(np.isfinite(resp)) and np.isfinite(ll)):
        fail(f"got non-finite output: responsibilities {resp}, log-likelihood {ll}. "
             "Subtract the row maximum inside the log-sum-exp.")
    close(resp, np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]), atol=1e-12, label="responsibilities")
