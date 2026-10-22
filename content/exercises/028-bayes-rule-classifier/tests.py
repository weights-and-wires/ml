"""Checks for the Bayes' rule classifier exercise."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from mlcheck import check, close, fail, shape


@check("normal_log_density_1d matches the log of the Gaussian density")
def test_normal_log_density(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    out = sub.normal_log_density_1d(x, 0.5, 2.0)
    shape(out, (5, 3))
    close(out, norm(0.5, np.sqrt(2.0)).logpdf(x), atol=1e-10)
    close(sub.normal_log_density_1d(np.array([3.0]), 1.5, 1.0), np.log([0.12951759566589174]))


@check("A density is a height, so a narrow bell has a positive log density")
def test_density_above_one(sub):
    out = float(sub.normal_log_density_1d(np.array(0.0), 0.0, 0.01))
    if out <= 0:
        fail(f"a Gaussian with variance 0.01 peaks at 3.99, so its log is about 1.38; got {out:.4f}")
    close(out, np.log(1.0 / np.sqrt(2 * np.pi * 0.01)), atol=1e-10)


@check("log_sum_exp agrees with scipy along either axis and drops that axis")
def test_lse_matches(sub):
    rng = np.random.default_rng(1)
    a = rng.normal(scale=3.0, size=(4, 6))
    for axis in (0, 1):
        out = sub.log_sum_exp(a, axis=axis)
        shape(out, (a.shape[1 - axis],))
        close(out, logsumexp(a, axis=axis), atol=1e-10)


@check("log_sum_exp does not underflow or overflow at large magnitudes")
def test_lse_stable(sub):
    for a in (np.array([[-1000.0, -1001.0]]), np.array([[1000.0, 999.0]])):
        out = sub.log_sum_exp(a, axis=1)
        if not np.all(np.isfinite(out)):
            fail(f"log_sum_exp({a.tolist()}) returned {out}; subtract the maximum first")
        close(out, logsumexp(a, axis=1), atol=1e-10)


@check("log_sum_exp subtracts a maximum per row, not one global maximum")
def test_lse_per_row(sub):
    a = np.array([[-1000.0, -1001.0], [1000.0, 999.0], [0.0, 0.0]])
    out = sub.log_sum_exp(a, axis=1)
    if not np.all(np.isfinite(out)):
        fail(f"log_sum_exp returned {out}; one row underflowed, so subtract the maximum of each row")
    close(out, logsumexp(a, axis=1), atol=1e-10)
    out0 = sub.log_sum_exp(a.T, axis=0)
    close(out0, logsumexp(a.T, axis=0), atol=1e-10, label="axis=0 on the transpose")


@check("Posterior rows add to 1 and reproduce the warehouse example")
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


@check("With equal densities the posterior equals the prior")
def test_equal_density(sub):
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
