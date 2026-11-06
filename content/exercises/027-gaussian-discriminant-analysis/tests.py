"""Checks for the Gaussian discriminant analysis exercise."""

import numpy as np
from scipy.stats import multivariate_normal

from mlcheck import check, close, fail, shape


def _three_class_data():
    rng = np.random.default_rng(3)
    shared = np.array([[2.0, 0.8, 0.0], [0.8, 1.0, 0.3], [0.0, 0.3, 0.5]])
    sizes = [30, 50, 20]
    centres = [np.zeros(3), np.array([3.0, 0.0, 1.0]), np.array([0.0, 4.0, -1.0])]
    X = np.vstack([rng.multivariate_normal(c, shared, size=m) for c, m in zip(centres, sizes)])
    y = np.repeat([0, 1, 2], sizes)
    return X, y


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


@check("gaussian_log_density matches the one-dimensional hand numbers")
def test_density_1d(sub):
    out = sub.gaussian_log_density(np.array([[3.0]]), np.array([2.0]), np.array([[1.0]]))
    shape(out, (1,))
    close(out, np.array([-1.4189385332]), atol=1e-8, label="class 0 at x = 3")
    out = sub.gaussian_log_density(np.array([[3.0]]), np.array([6.0]), np.array([[1.0]]))
    close(out, np.array([-5.4189385332]), atol=1e-8, label="class 1 at x = 3")


@check("gaussian_log_density agrees with scipy for a tilted covariance")
def test_density_nd(sub):
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
