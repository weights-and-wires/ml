"""Checks for the Gaussian mixtures and EM exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def two_points():
    X = np.array([[0.0], [2.0]])
    means = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])
    return X, means, covs, weights


def blobs(seed: int = 3, n_per: int = 20):
    rng = np.random.default_rng(seed)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    return np.concatenate([s + 0.8 * rng.normal(size=(n_per, 2)) for s in spots])


HARD_RESP = np.array([[0.880797, 0.119203], [0.119203, 0.880797]])


@check("log_gaussian gives -0.918939 at the peak of a standard normal, matches a tilted 2-D formula, and sums 1-D log densities when diagonal")
def test_log_gaussian(sub):
    f = has(sub, "log_gaussian")
    got = np.asarray(f(np.array([[0.0], [2.0]]), np.array([0.0]), np.array([[1.0]])))
    shape(got, (2,), label="log densities")
    close(got, np.array([-0.5 * np.log(2 * np.pi), -0.5 * np.log(2 * np.pi) - 2.0]), atol=1e-12,
          label="log densities at 0 and 2")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 2)) * 2
    mean = np.array([0.5, -1.0])
    cov = np.array([[2.0, 1.2], [1.2, 1.0]])
    want = np.empty(25)
    inv = np.linalg.inv(cov)
    for i in range(25):
        g = X[i] - mean
        want[i] = -0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(cov)) + g @ inv @ g)
    close(np.asarray(f(X, mean, cov)), want, atol=1e-10, label="tilted log density")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    mean = np.array([0.1, 0.2, -0.3])
    var = np.array([0.5, 2.0, 1.5])
    want = np.sum(-0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var), axis=1)
    close(np.asarray(f(X, mean, np.diag(var))), want, atol=1e-10, label="diagonal log density")


@check("logsumexp_rows survives scores near -1000, matches the naive formula on small ones, and leaves its input alone")
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


@check("gmm_e_step reproduces shares 0.880797 and 0.119203 with log L = -2.970, follows Bayes' rule, and stays finite far from every blob")
def test_e_step(sub):
    e_step = has(sub, "gmm_e_step")
    X, means, covs, weights = two_points()
    resp, ll = e_step(X, means, covs, weights)
    shape(resp, (2, 2), label="responsibilities")
    close(resp, HARD_RESP, atol=1e-6, label="responsibilities")
    if not isinstance(ll, float):
        fail(f"expected a Python float for the log-likelihood, got {type(ll).__name__}")
    almost(ll, -2.970315, tol=1e-6, label="log-likelihood")
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
    close(resp, np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]), atol=1e-12, label="responsibilities far from every blob")


@check("gmm_m_step on the two points gives weights 0.5, means 0.2384 and 1.7616, variance 0.42")
def test_m_step_hand(sub):
    m_step = has(sub, "gmm_m_step")
    X = np.array([[0.0], [2.0]])
    means, covs, weights = m_step(X, HARD_RESP, 0.0)
    shape(means, (2, 1), label="means")
    shape(covs, (2, 1, 1), label="covs")
    shape(weights, (2,), label="weights")
    close(weights, np.array([0.5, 0.5]), atol=1e-9, label="weights")
    close(means.ravel(), np.array([0.238406, 1.761594]), atol=1e-6, label="means")
    close(covs.ravel(), np.array([0.419974, 0.419974]), atol=1e-6, label="variances")


@check("gmm_m_step with hard shares returns the plain mean and covariance plus reg_covar, and with soft shares measures around the new mean")
def test_m_step_general(sub):
    m_step = has(sub, "gmm_m_step")
    rng = np.random.default_rng(6)
    X = rng.normal(loc=2.0, size=(100, 2))
    resp = np.zeros((100, 2))
    resp[:50, 0] = 1.0
    resp[50:, 1] = 1.0
    means, covs, weights = m_step(X, resp, 0.0)
    close(weights, np.array([0.5, 0.5]), atol=1e-12, label="weights")
    close(means[0], X[:50].mean(axis=0), atol=1e-9, label="mean of the first half")
    close(covs[1], np.cov(X[50:].T, bias=True), atol=1e-9, label="covariance of the second half")
    _, covs_reg, _ = m_step(X, resp, 0.25)
    close(covs_reg[0] - covs[0], 0.25 * np.eye(2), atol=1e-9, label="the reg_covar term")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3)) * np.array([1.0, 2.0, 0.5])
    resp = rng.dirichlet(np.ones(3), size=60)
    means, covs, weights = m_step(X, resp, 0.0)
    for j in range(3):
        want_mean = np.average(X, axis=0, weights=resp[:, j])
        close(means[j], want_mean, atol=1e-9, label=f"mean of blob {j}")
        want_cov = np.cov(X.T, aweights=resp[:, j], ddof=0)
        close(covs[j], want_cov, atol=1e-9, label=f"covariance of blob {j}")
    close(weights, resp.sum(axis=0) / 60, atol=1e-12, label="weights")


@check("gmm_elbo gives -2.970 at the posterior and -2.626 after the M-step")
def test_elbo_hand(sub):
    elbo = has(sub, "gmm_elbo")
    e_step = has(sub, "gmm_e_step")
    m_step = has(sub, "gmm_m_step")
    X, means, covs, weights = two_points()
    resp, ll = e_step(X, means, covs, weights)
    got = elbo(X, resp, means, covs, weights)
    if not isinstance(got, float):
        fail(f"expected a Python float, got {type(got).__name__}")
    almost(got, ll, tol=1e-9, label="ELBO when the shares are the posterior")
    means2, covs2, weights2 = m_step(X, resp, 0.0)
    almost(elbo(X, resp, means2, covs2, weights2), -2.625942, tol=1e-5, label="ELBO after the M-step")


@check("gmm_elbo never exceeds log L for shares that are not the posterior, and stays finite when a share is exactly 0")
def test_elbo_bound(sub):
    elbo = has(sub, "gmm_elbo")
    e_step = has(sub, "gmm_e_step")
    X = blobs(seed=4, n_per=10)
    means = np.array([[0.5, 0.0], [5.0, 0.5], [3.0, 4.0]])
    covs = np.stack([np.eye(2) * 1.5, np.eye(2), np.array([[1.0, 0.4], [0.4, 0.8]])])
    weights = np.array([0.3, 0.3, 0.4])
    resp, ll = e_step(X, means, covs, weights)
    rng = np.random.default_rng(0)
    for _ in range(5):
        other = rng.dirichlet(np.ones(3), size=len(X))
        value = elbo(X, other, means, covs, weights)
        if value > ll + 1e-9:
            fail(f"the bound {value:.4f} is above log L = {ll:.4f}")
    almost(elbo(X, resp, means, covs, weights), ll, tol=1e-8, label="bound at the posterior")
    X = np.array([[0.0], [10.0]])
    means = np.array([[0.0], [10.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])
    resp = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = elbo(X, resp, means, covs, weights)
    if not np.isfinite(got):
        fail(f"got {got}. A share of 0 contributes 0 to the bound, but 0 * log(0) is nan.")
    almost(got, 2 * (np.log(0.5) - 0.5 * np.log(2 * np.pi)), tol=1e-9, label="bound with a zero share")


@check("fit_gmm returns a log-likelihood history of length iters + 1 that never falls")
def test_fit_history(sub):
    fit = has(sub, "fit_gmm")
    X = blobs(seed=5)
    means, covs, weights, resp, history = fit(X, 3, 12, np.random.default_rng(0))
    shape(np.asarray(means), (3, 2), label="means")
    shape(np.asarray(covs), (3, 2, 2), label="covs")
    shape(np.asarray(resp), (60, 3), label="responsibilities")
    if len(history) != 13:
        fail(f"expected 13 history entries (the first E-step plus 12 rounds), got {len(history)}")
    # history[0] is log L at the starting blobs, before any M-step has run.
    seeds = has(sub, "kmeans_plusplus_init")(X, 3, np.random.default_rng(0))
    start_covs = np.stack([np.var(X, axis=0).mean() * np.eye(2)] * 3)
    _, start_ll = has(sub, "gmm_e_step")(X, seeds, start_covs, np.full(3, 1 / 3))
    almost(history[0], start_ll, tol=1e-6, label="log L before the first M-step")
    if np.any(np.diff(history) < -1e-6):
        fail(f"log L fell on some round, which a correct E-step then M-step cannot do: {[round(h, 2) for h in history]}")
    if history[-1] <= history[0] + 1.0:
        fail(f"log L barely moved: {history[0]:.2f} to {history[-1]:.2f}")


@check("fit_gmm recovers the three planted blobs")
def test_fit_recovers(sub):
    fit = has(sub, "fit_gmm")
    X = blobs(seed=5)
    best = None
    for seed in range(4):
        out = fit(X, 3, 40, np.random.default_rng(seed))
        if best is None or out[4][-1] > best[4][-1]:
            best = out
    means, _, weights, _, _ = best
    truth = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    for t in truth:
        gap = np.min(np.linalg.norm(means - t, axis=1))
        if gap > 0.6:
            fail(f"no fitted mean lies within 0.6 of the planted centre {t}; the nearest is {gap:.2f} away")
    close(np.sort(weights), np.full(3, 1 / 3), atol=0.08, label="weights")
