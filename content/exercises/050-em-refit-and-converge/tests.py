"""Checks for the EM exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def blobs(seed: int = 3, n_per: int = 20):
    rng = np.random.default_rng(seed)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    return np.concatenate([s + 0.8 * rng.normal(size=(n_per, 2)) for s in spots])


def two_points():
    X = np.array([[0.0], [2.0]])
    resp = np.array([[0.880797, 0.119203], [0.119203, 0.880797]])
    return X, resp


@check("gmm_m_step on the chapter's two points gives weights 0.5, means 0.2384 and 1.7616, variance 0.42")
def test_m_step_hand(sub):
    m_step = has(sub, "gmm_m_step")
    X, resp = two_points()
    means, covs, weights = m_step(X, resp, 0.0)
    shape(means, (2, 1), label="means")
    shape(covs, (2, 1, 1), label="covs")
    shape(weights, (2,), label="weights")
    close(weights, np.array([0.5, 0.5]), atol=1e-9, label="weights")
    close(means.ravel(), np.array([0.238406, 1.761594]), atol=1e-6, label="means")
    close(covs.ravel(), np.array([0.419974, 0.419974]), atol=1e-6, label="variances")


@check("gmm_m_step with all-or-nothing shares returns the plain mean and covariance, plus reg_covar")
def test_m_step_hard(sub):
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


@check("gmm_m_step measures the covariance around the new mean, with soft shares")
def test_m_step_soft(sub):
    m_step = has(sub, "gmm_m_step")
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


@check("gmm_elbo gives -2.970 at the posterior and -2.626 after the chapter's M-step")
def test_elbo_hand(sub):
    elbo = has(sub, "gmm_elbo")
    e_step = has(sub, "gmm_e_step")
    m_step = has(sub, "gmm_m_step")
    X, _ = two_points()
    means = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])
    resp, ll = e_step(X, means, covs, weights)
    got = elbo(X, resp, means, covs, weights)
    if not isinstance(got, float):
        fail(f"expected a Python float, got {type(got).__name__}")
    almost(got, ll, tol=1e-9, label="ELBO when the shares are the posterior")
    means2, covs2, weights2 = m_step(X, resp, 0.0)
    almost(elbo(X, resp, means2, covs2, weights2), -2.625942, tol=1e-5, label="ELBO after the M-step")


@check("gmm_elbo never exceeds log L, for shares that are not the posterior")
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


@check("gmm_elbo stays finite when a share is exactly 0")
def test_elbo_zero_share(sub):
    elbo = has(sub, "gmm_elbo")
    X = np.array([[0.0], [10.0]])
    means = np.array([[0.0], [10.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])
    resp = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = elbo(X, resp, means, covs, weights)
    if not np.isfinite(got):
        fail(f"got {got}. A share of 0 contributes 0 to the bound, but 0 * log(0) is nan.")
    almost(got, 2 * (np.log(0.5) - 0.5 * np.log(2 * np.pi)), tol=1e-9, label="bound")


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
