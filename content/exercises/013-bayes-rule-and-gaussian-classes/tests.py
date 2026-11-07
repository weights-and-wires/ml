"""Checks for the Bayes' rule and Gaussian classes exercise."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from mlcheck import check, close, fail, shape


def _three_class_data():
    rng = np.random.default_rng(3)
    shared = np.array([[2.0, 0.8, 0.0], [0.8, 1.0, 0.3], [0.0, 0.3, 0.5]])
    sizes = [30, 50, 20]
    centres = [np.zeros(3), np.array([3.0, 0.0, 1.0]), np.array([0.0, 4.0, -1.0])]
    X = np.vstack([rng.multivariate_normal(c, shared, size=m) for c, m in zip(centres, sizes)])
    y = np.repeat([0, 1, 2], sizes)
    return X, y


@check("normal_log_density_1d matches the log of the Gaussian density, and a narrow bell has a positive log density")
def test_normal_log_density(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    out = sub.normal_log_density_1d(x, 0.5, 2.0)
    shape(out, (5, 3))
    close(out, norm(0.5, np.sqrt(2.0)).logpdf(x), atol=1e-10)
    close(sub.normal_log_density_1d(np.array([3.0]), 1.5, 1.0), np.log([0.12951759566589174]))
    out = float(sub.normal_log_density_1d(np.array(0.0), 0.0, 0.01))
    if out <= 0:
        fail(f"a Gaussian with variance 0.01 peaks at 3.99, so its log is about 1.38; got {out:.4f}")
    close(out, np.log(1.0 / np.sqrt(2 * np.pi * 0.01)), atol=1e-10)


@check("log_sum_exp agrees with scipy along either axis, and stays finite at large magnitudes in every row")
def test_lse_matches(sub):
    rng = np.random.default_rng(1)
    a = rng.normal(scale=3.0, size=(4, 6))
    for axis in (0, 1):
        out = sub.log_sum_exp(a, axis=axis)
        shape(out, (a.shape[1 - axis],))
        close(out, logsumexp(a, axis=axis), atol=1e-10)
    a = np.array([[-1000.0, -1001.0], [1000.0, 999.0], [0.0, 0.0]])
    out = sub.log_sum_exp(a, axis=1)
    if not np.all(np.isfinite(out)):
        fail(f"log_sum_exp returned {out}; one row underflowed, so subtract the maximum of each row")
    close(out, logsumexp(a, axis=1), atol=1e-10)
    out0 = sub.log_sum_exp(a.T, axis=0)
    close(out0, logsumexp(a.T, axis=0), atol=1e-10, label="axis=0 on the transpose")


@check("log_posterior_from_log_density reproduces the warehouse example, adds to 1, and returns the prior for equal densities")
def test_posterior(sub):
    log_density = np.log(np.array([[0.12951759566589174, 0.24197072451914337]]))
    out = sub.log_posterior_from_log_density(log_density, np.array([0.9, 0.1]))
    shape(out, (1, 2))
    close(np.exp(out), np.array([[0.8281005081, 0.1718994919]]), atol=1e-8)
    rng = np.random.default_rng(2)
    big = rng.normal(scale=5.0, size=(7, 4)) - 800.0
    post = sub.log_posterior_from_log_density(big, np.array([0.1, 0.2, 0.3, 0.4]))
    if not np.all(np.isfinite(post)):
        fail("log posteriors are not finite when the log densities are near -800")
    close(np.exp(post).sum(axis=1), np.ones(7), atol=1e-10)
    log_density = np.full((3, 3), -2.0)
    priors = np.array([0.7, 0.2, 0.1])
    out = np.exp(sub.log_posterior_from_log_density(log_density, priors))
    close(out, np.tile(priors, (3, 1)), atol=1e-12)


@check("The prior flips the winner at 3 kg")
def test_prior_flips(sub):
    x = np.array([3.0])
    log_density = np.stack(
        [sub.normal_log_density_1d(x, 1.5, 1.0), sub.normal_log_density_1d(x, 4.0, 1.0)], axis=1
    )
    even = sub.predict_class(sub.log_posterior_from_log_density(log_density, np.array([0.5, 0.5])))
    tilted = sub.predict_class(sub.log_posterior_from_log_density(log_density, np.array([0.9, 0.1])))
    if int(even[0]) != 1:
        fail(f"with equal priors the heavier warehouse (1) should win at 3 kg; got {int(even[0])}")
    if int(tilted[0]) != 0:
        fail(f"with a 90/10 prior the common warehouse (0) should win at 3 kg; got {int(tilted[0])}")


@check("fit_gda reproduces the four-point example from the chapter")
def test_fit_hand(sub):
    X = np.array([[1.0], [3.0], [5.0], [7.0]])
    y = np.array([0, 0, 1, 1])
    means, cov, priors = sub.fit_gda(X, y)
    shape(means, (2, 1))
    shape(cov, (1, 1))
    close(means.ravel(), np.array([2.0, 6.0]))
    close(cov, np.array([[1.0]]), label="shared covariance (divide the pooled sum by n = 4)")
    close(priors, np.array([0.5, 0.5]))


@check("fit_gda: means, priors, and a pooled covariance from each point's own class mean")
def test_fit_general(sub):
    X, y = _three_class_data()
    means, cov, priors = sub.fit_gda(X, y)
    shape(means, (3, 3))
    shape(cov, (3, 3))
    close(priors, np.array([0.3, 0.5, 0.2]))
    want_means = np.array([X[y == k].mean(axis=0) for k in range(3)])
    close(means, want_means, atol=1e-10)
    centred = X - want_means[y]
    close(cov, centred.T @ centred / len(X), atol=1e-10, label="covariance")
    close(cov, cov.T, atol=1e-12, label="covariance symmetry")


@check("gaussian_log_density matches the one-dimensional hand numbers and scipy for a tilted covariance")
def test_density_1d(sub):
    out = sub.gaussian_log_density(np.array([[3.0]]), np.array([2.0]), np.array([[1.0]]))
    shape(out, (1,))
    close(out, np.array([-1.4189385332]), atol=1e-8, label="class 0 at x = 3")
    out = sub.gaussian_log_density(np.array([[3.0]]), np.array([6.0]), np.array([[1.0]]))
    close(out, np.array([-5.4189385332]), atol=1e-8, label="class 1 at x = 3")
    rng = np.random.default_rng(4)
    cov = np.array([[4.0, 2.0], [2.0, 3.0]])
    mean = np.array([1.0, -1.0])
    X = rng.normal(scale=2.0, size=(10, 2))
    out = sub.gaussian_log_density(X, mean, cov)
    shape(out, (10,))
    close(out, multivariate_normal(mean, cov).logpdf(X), atol=1e-9)


@check("Equal straight-line distance does not mean equal density")
def test_mahalanobis(sub):
    cov = np.array([[4.0, 0.0], [0.0, 0.25]])
    along_long = np.array([[2.0, 0.0]])
    along_short = np.array([[0.0, 2.0]])
    a = sub.gaussian_log_density(along_long, np.zeros(2), cov)[0]
    b = sub.gaussian_log_density(along_short, np.zeros(2), cov)[0]
    if abs((a - b) - 7.5) > 1e-8:
        fail(f"the squared Mahalanobis distances are 1 and 16, so the log densities differ by 7.5; got a gap of {a - b:.4f}")


@check("gda_log_posterior reproduces the posterior 0.982 and 0.018 at x = 3")
def test_posterior_hand(sub):
    means = np.array([[2.0], [6.0]])
    out = sub.gda_log_posterior(np.array([[3.0]]), means, np.array([[1.0]]), np.array([0.5, 0.5]))
    shape(out, (1, 2))
    close(np.exp(out), np.array([[0.9820137900, 0.0179862100]]), atol=1e-8)


@check("gda_log_posterior matches an independent computation on three unequal classes")
def test_posterior_general(sub):
    X, y = _three_class_data()
    means, cov, priors = sub.fit_gda(X, y)
    out = sub.gda_log_posterior(X, means, cov, priors)
    shape(out, (len(X), 3))
    close(np.exp(out).sum(axis=1), np.ones(len(X)), atol=1e-10, label="row sums")
    joint = np.stack(
        [np.log(priors[k]) + multivariate_normal(means[k], cov).logpdf(X) for k in range(3)], axis=1
    )
    want = joint - np.log(np.exp(joint).sum(axis=1, keepdims=True))
    close(out, want, atol=1e-8)
    if (out.argmax(axis=1) == y).mean() < 0.9:
        fail("a well-separated three-class problem should be classified correctly at least 90% of the time")

