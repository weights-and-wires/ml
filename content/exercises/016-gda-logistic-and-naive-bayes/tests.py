"""Checks for the GDA, logistic regression, and naive Bayes exercise."""

import numpy as np
from scipy.stats import norm

from mlcheck import check, close, fail, shape

DOCS = np.array(
    [[2, 1, 1, 0, 0, 0], [1, 2, 0, 0, 1, 0], [0, 0, 0, 2, 1, 1], [1, 0, 0, 1, 2, 0]]
)
LABELS = np.array([1, 1, 0, 0])  # class 1 is spam, class 0 is ham


def _two_class_problem():
    rng = np.random.default_rng(5)
    cov = np.array([[2.0, 0.9, 0.0], [0.9, 1.0, 0.4], [0.0, 0.4, 0.8]])
    means = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 1.0]])
    priors = np.array([0.7, 0.3])
    X = rng.multivariate_normal(np.zeros(3), 1.5 * cov, size=40)
    return X, means, cov, priors


@check("gda_to_logistic gives w = 4 and b = -16 on the four-point example")
def test_hand_example(sub):
    w, b = sub.gda_to_logistic(np.array([[2.0], [6.0]]), np.array([[1.0]]), np.array([0.5, 0.5]))
    shape(np.asarray(w), (1,))
    close(w, np.array([4.0]), label="w")
    close(float(b), -16.0, label="b")
    if not isinstance(b, float):
        fail(f"b should be a Python float, got {type(b).__name__}")


@check("sigmoid(x @ w + b) reproduces GDA's class 1 probability with unequal priors, and w is the shape-weighted mean difference")
def test_matches_gda(sub):
    X, means, cov, priors = _two_class_problem()
    w, b = sub.gda_to_logistic(means, cov, priors)
    want = np.exp(sub.gda_log_posterior(X, means, cov, priors))[:, 1]
    close(sub.sigmoid(X @ w + b), want, atol=1e-9)
    _, means, cov, priors = _two_class_problem()
    w, _ = sub.gda_to_logistic(means, cov, priors)
    close(w, np.linalg.solve(cov, means[1] - means[0]), atol=1e-9)
    if np.allclose(w, means[1] - means[0], atol=1e-3):
        fail("w equals the plain mean difference; multiply by the inverse covariance")


@check("log_odds is log p(1|x) - log p(0|x) and is a straight line in x")
def test_log_odds(sub):
    X, means, cov, priors = _two_class_problem()
    out = sub.log_odds(X, means, cov, priors)
    shape(out, (len(X),))
    w = np.linalg.solve(cov, means[1] - means[0])
    b = np.log(priors[1] / priors[0]) - 0.5 * (means[0] + means[1]) @ w
    close(out, X @ w + b, atol=1e-8, label="log odds against w.x + b")
    a, d = X[0], X[1] - X[0]
    line = sub.log_odds(np.stack([a, a + d, a + 2 * d]), means, cov, priors)
    close(line[0] - 2 * line[1] + line[2], 0.0, atol=1e-8, label="second difference along a line")


@check("decision_boundary_1d is x = 4 for the hand example, and a rarer class pushes the boundary toward itself")
def test_boundary_hand(sub):
    x = sub.decision_boundary_1d(np.array([2.0, 6.0]), 1.0, np.array([0.5, 0.5]))
    close(float(x), 4.0)
    x = sub.decision_boundary_1d(np.array([[2.0], [6.0]]), 4.0, np.array([0.5, 0.5]))
    close(float(x), 4.0, label="equal priors put the boundary at the midpoint whatever the variance")
    means = np.array([[2.0], [6.0]])
    priors = np.array([0.9, 0.1])
    x = float(sub.decision_boundary_1d(means, 1.0, priors))
    close(x, 4.0 + np.log(9.0) / 4.0, atol=1e-9)
    odds = sub.log_odds(np.array([[x]]), means, np.array([[1.0]]), priors)
    close(odds, np.array([0.0]), atol=1e-9, label="log odds at the boundary")
    wide = float(sub.decision_boundary_1d(means, 4.0, priors))
    close(wide, 4.0 + np.log(9.0), atol=1e-9, label="variance 4 (a standard deviation of 2) moves the boundary four times as far")
    odds = sub.log_odds(np.array([[wide]]), means, np.array([[4.0]]), priors)
    close(odds, np.array([0.0]), atol=1e-9, label="log odds at the wide boundary")


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


@check("fit_naive_bayes_gaussian gives per-class, per-feature estimates with unequal classes, and a constant feature gets the variance floor")
def test_gaussian_general(sub):
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([0, 5], [1.0, 2.0], size=(30, 2)), rng.normal([3, 0], [0.5, 1.0], size=(10, 2))])
    y = np.repeat([0, 1], [30, 10])
    means, variances, priors = sub.fit_naive_bayes_gaussian(X, y)
    for k in range(2):
        close(means[k], X[y == k].mean(axis=0), atol=1e-10, label=f"means[{k}]")
        close(variances[k], X[y == k].var(axis=0), atol=1e-10, label=f"variances[{k}]")
    close(priors, np.array([0.75, 0.25]))
    X = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 9.0], [7.0, 9.0]])
    y = np.array([0, 0, 1, 1])
    _, variances, _ = sub.fit_naive_bayes_gaussian(X, y, var_floor=1e-3)
    close(variances[:, 1], np.array([1e-3, 1e-3]), label="variance of the constant feature")
    close(variances[:, 0], np.array([1.0, 1.0]), label="variance of the ordinary feature")


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


@check("fit_naive_bayes_multinomial reproduces the smoothed word probabilities, adds alpha once per word, and leaves no zero")
def test_multinomial_fit(sub):
    log_theta, log_priors = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    shape(log_theta, (2, 6))
    shape(log_priors, (2,))
    close(np.exp(log_theta[1]), np.array([4, 4, 2, 1, 2, 1]) / 14.0, label="spam word probabilities")
    close(np.exp(log_theta[0]), np.array([2, 1, 1, 4, 4, 2]) / 14.0, label="ham word probabilities")
    close(np.exp(log_priors), np.array([0.5, 0.5]))
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


@check("multinomial_nb_log_posterior: a word seen twice votes twice, and a 90/10 prior shifts the odds by log 9")
def test_counts_vote(sub):
    log_theta, log_priors = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    once = sub.multinomial_nb_log_posterior(np.array([[1, 0, 0, 0, 0, 0]]), log_theta, log_priors)
    twice = sub.multinomial_nb_log_posterior(np.array([[2, 0, 0, 0, 0, 0]]), log_theta, log_priors)
    odds_once = once[0, 1] - once[0, 0]
    odds_twice = twice[0, 1] - twice[0, 0]
    close(odds_twice, 2 * odds_once, atol=1e-9, label="log odds with two copies of the word")
    many = np.random.default_rng(2).integers(0, 5, size=(6, 6))
    close(np.exp(sub.multinomial_nb_log_posterior(many, log_theta, log_priors)).sum(axis=1), np.ones(6), atol=1e-10)
    log_theta, _ = sub.fit_naive_bayes_multinomial(DOCS, LABELS, alpha=1.0)
    doc = np.array([[1, 1, 1, 1, 1, 1]])
    even = sub.multinomial_nb_log_posterior(doc, log_theta, np.log(np.array([0.5, 0.5])))
    tilted = sub.multinomial_nb_log_posterior(doc, log_theta, np.log(np.array([0.1, 0.9])))
    shift = (tilted[0, 1] - tilted[0, 0]) - (even[0, 1] - even[0, 0])
    close(shift, np.log(9.0), atol=1e-9, label="change in log odds when the spam prior goes from 0.5 to 0.9")

