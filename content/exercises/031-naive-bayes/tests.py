"""Checks for the naive Bayes exercise."""

import numpy as np
from scipy.stats import norm

from mlcheck import check, close, fail, shape

DOCS = np.array(
    [[2, 1, 1, 0, 0, 0], [1, 2, 0, 0, 1, 0], [0, 0, 0, 2, 1, 1], [1, 0, 0, 1, 2, 0]]
)
LABELS = np.array([1, 1, 0, 0])  # class 1 is spam, class 0 is ham


@check("fit_naive_bayes_gaussian: variance 1 for the values 1 and 3, not 0.5")
def test_gaussian_hand(sub):
    X = np.array([[1.0], [3.0], [5.0], [7.0]])
    y = np.array([0, 0, 1, 1])
    means, variances, priors = sub.fit_naive_bayes_gaussian(X, y)
    shape(means, (2, 1))
    shape(variances, (2, 1))
    close(means.ravel(), np.array([2.0, 6.0]))
    close(variances.ravel(), np.array([1.0, 1.0]), label="variances (divide by the class count, not by n)")
    close(priors, np.array([0.5, 0.5]))


@check("fit_naive_bayes_gaussian: per-class, per-feature estimates with unequal classes")
def test_gaussian_general(sub):
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([0, 5], [1.0, 2.0], size=(30, 2)), rng.normal([3, 0], [0.5, 1.0], size=(10, 2))])
    y = np.repeat([0, 1], [30, 10])
    means, variances, priors = sub.fit_naive_bayes_gaussian(X, y)
    for k in range(2):
        close(means[k], X[y == k].mean(axis=0), atol=1e-10, label=f"means[{k}]")
        close(variances[k], X[y == k].var(axis=0), atol=1e-10, label=f"variances[{k}]")
    close(priors, np.array([0.75, 0.25]))


@check("A constant feature gets the variance floor, not zero")
def test_gaussian_floor(sub):
    X = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 9.0], [7.0, 9.0]])
    y = np.array([0, 0, 1, 1])
    _, variances, _ = sub.fit_naive_bayes_gaussian(X, y, var_floor=1e-3)
    close(variances[:, 1], np.array([1e-3, 1e-3]), label="variance of the constant feature")
    close(variances[:, 0], np.array([1.0, 1.0]), label="variance of the ordinary feature")


@check("fit_naive_bayes_multinomial reproduces the smoothed word probabilities")
def test_multinomial_fit(sub):
    log_theta, log_priors = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    shape(log_theta, (2, 6))
    shape(log_priors, (2,))
    close(np.exp(log_theta[1]), np.array([4, 4, 2, 1, 2, 1]) / 14.0, label="spam word probabilities")
    close(np.exp(log_theta[0]), np.array([2, 1, 1, 4, 4, 2]) / 14.0, label="ham word probabilities")
    close(np.exp(log_priors), np.array([0.5, 0.5]))


@check("Smoothing adds alpha once per word in the denominator, and no row has a zero")
def test_multinomial_alpha(sub):
    log_theta, _ = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=0.1)
    if not np.all(np.isfinite(log_theta)):
        fail("log_theta has an infinite entry; every word needs a probability above zero")
    close(np.exp(log_theta).sum(axis=1), np.ones(2), atol=1e-12, label="row sums")
    close(np.exp(log_theta[1]), (np.array([3, 3, 1, 0, 1, 0]) + 0.1) / (8 + 0.6), label="spam row with alpha = 0.1")


@check("multinomial_nb_log_posterior gives 0.80 spam for 'free free money meeting'")
def test_multinomial_posterior(sub):
    log_theta, log_priors = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    out = sub.multinomial_nb_log_posterior(np.array([[2, 1, 0, 1, 0, 0]]), log_theta, log_priors)
    shape(out, (1, 2))
    close(np.exp(out), np.array([[0.2, 0.8]]), atol=1e-9)


@check("A word seen twice votes twice, and rows exponentiate to a distribution")
def test_counts_vote(sub):
    log_theta, log_priors = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    once = sub.multinomial_nb_log_posterior(np.array([[1, 0, 0, 0, 0, 0]]), log_theta, log_priors)
    twice = sub.multinomial_nb_log_posterior(np.array([[2, 0, 0, 0, 0, 0]]), log_theta, log_priors)
    odds_once = once[0, 1] - once[0, 0]
    odds_twice = twice[0, 1] - twice[0, 0]
    close(odds_twice, 2 * odds_once, atol=1e-9, label="log odds with two copies of the word")
    many = np.random.default_rng(2).integers(0, 5, size=(6, 6))
    close(np.exp(sub.multinomial_nb_log_posterior(many, log_theta, log_priors)).sum(axis=1), np.ones(6), atol=1e-10)


@check("multinomial_nb_log_posterior adds the log prior once, so a 90/10 prior shifts the odds by log 9")
def test_log_prior_shift(sub):
    log_theta, _ = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    doc = np.array([[1, 1, 1, 1, 1, 1]])
    even = sub.multinomial_nb_log_posterior(doc, log_theta, np.log(np.array([0.5, 0.5])))
    tilted = sub.multinomial_nb_log_posterior(doc, log_theta, np.log(np.array([0.1, 0.9])))
    shift = (tilted[0, 1] - tilted[0, 0]) - (even[0, 1] - even[0, 0])
    close(shift, np.log(9.0), atol=1e-9, label="change in log odds when the spam prior goes from 0.5 to 0.9")


@check("Fitted Gaussian naive Bayes scores points like a sum of one-dimensional Gaussians")
def test_gaussian_posterior(sub):
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal([0, 0, 0], 1.0, size=(40, 3)), rng.normal([2, 1, 0], 1.5, size=(20, 3))])
    y = np.repeat([0, 1], [40, 20])
    means, variances, priors = sub.fit_naive_bayes_gaussian(X, y)
    out = sub.gaussian_nb_log_posterior(X, means, variances, priors)
    joint = np.stack(
        [np.log(priors[k]) + norm(means[k], np.sqrt(variances[k])).logpdf(X).sum(axis=1) for k in range(2)],
        axis=1,
    )
    want = joint - np.log(np.exp(joint).sum(axis=1, keepdims=True))
    close(out, want, atol=1e-8)
    if (out.argmax(axis=1) == y).mean() < 0.8:
        fail("training accuracy below 0.8 on two well-separated clouds")
