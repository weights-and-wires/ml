"""Checks for the exercise on GDA as a logistic regression."""

import numpy as np

from mlcheck import check, close, fail, shape


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


@check("sigmoid(x @ w + b) reproduces GDA's class 1 probability, with unequal priors")
def test_matches_gda(sub):
    X, means, cov, priors = _two_class_problem()
    w, b = sub.gda_to_logistic(means, cov, priors)
    want = np.exp(sub.gda_log_posterior(X, means, cov, priors))[:, 1]
    close(sub.sigmoid(X @ w + b), want, atol=1e-9)


@check("w follows the shape-weighted mean difference, not the plain difference")
def test_w_direction(sub):
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


@check("decision_boundary_1d is x = 4 for the hand example")
def test_boundary_hand(sub):
    x = sub.decision_boundary_1d(np.array([2.0, 6.0]), 1.0, np.array([0.5, 0.5]))
    close(float(x), 4.0)
    x = sub.decision_boundary_1d(np.array([[2.0], [6.0]]), 4.0, np.array([0.5, 0.5]))
    close(float(x), 4.0, label="equal priors put the boundary at the midpoint whatever the variance")


@check("A rarer class pushes the boundary toward itself, and the log odds vanish there")
def test_boundary_prior(sub):
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
