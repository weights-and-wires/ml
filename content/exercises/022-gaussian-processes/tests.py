"""Checks for the Gaussian process exercise.

Four plausible wrong implementations are named below. Each check that would
accept one of them fails it, and shows the expected value and the value it
returns:

- posterior_forgets_subtraction returns the prior variance k(x*, x*) and drops
  the k_*^T (K + noise I)^{-1} k_* term. The variance check at training points
  catches it, because the true variance there is about noise / (1 + noise).
- prior_samples_use_L_not_Lt returns z @ L instead of z @ L.T, so the sample
  covariance is L^T L instead of K. The sample-covariance check catches it.
- lml_without_logdet drops the 0.5 log det term. The marginal likelihood checks
  catch it, because the value is off by 0.5 log det(K + noise I).
- gradient_in_length_scale differentiates with respect to the length scale
  rather than its log, so it is off by the factor length_scale. The
  finite-difference check at two length scales catches it.
"""

import numpy as np
from scipy.stats import multivariate_normal

from mlcheck import almost, check, close, fail, finite, finite_difference, has, shape


def ref_rbf(A, B, ls):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2.0 * ls**2))


def ref_lml(X, y, ls, noise):
    A = ref_rbf(X, X, ls) + noise * np.eye(len(y))
    eigvals = np.linalg.eigvalsh(A)
    return -0.5 * y @ np.linalg.solve(A, y) - 0.5 * np.log(eigvals).sum() - 0.5 * len(y) * np.log(2.0 * np.pi)


def two_points():
    X = np.array([[0.0], [1.0]])
    y = np.array([1.0, 2.0])
    return X, y, lambda a, b: ref_rbf(a, b, 1.0)


@check("rbf_kernel: ones on the diagonal, exp(-1/2) at distance 1, and the length scale rescales distance")
def test_rbf(sub):
    fn = has(sub, "rbf_kernel")
    X = np.array([[0.0], [1.0]])
    K = fn(X, X, 1.0)
    shape(K, (2, 2), label="K")
    close(K, np.array([[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]]), atol=1e-12, label="K for length scale 1")
    close(fn(np.array([[0.0]]), np.array([[2.0]]), 2.0), np.array([[np.exp(-0.5)]]), atol=1e-12,
          label="distance 2 at length scale 2")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    B = rng.normal(size=(4, 3))
    got = fn(A, B, 0.8)
    shape(got, (5, 4), label="k(A, B)")
    close(got, ref_rbf(A, B, 0.8), atol=1e-10, label="k(A, B) in 3 dimensions")
    close(np.diag(fn(A, A, 0.8)), np.ones(5), atol=1e-12, label="diagonal of k(A, A)")


@check("gp_prior_samples: the right shape, and the sample covariance of many draws matches K")
def test_prior_covariance(sub):
    fn = has(sub, "gp_prior_samples")
    X = np.array([[0.0], [0.5], [1.5], [3.0]])
    kern = lambda a, b: ref_rbf(a, b, 1.0)  # noqa: E731
    f = fn(X, kern, 40000, np.random.default_rng(0))
    shape(f, (40000, 4), label="samples")
    finite(f, label="samples")
    close(f.mean(axis=0), np.zeros(4), atol=0.03, label="sample mean")
    cov = (f.T @ f) / f.shape[0]
    close(cov, kern(X, X), atol=0.04, label="sample covariance against K")


@check("gp_prior_samples: a dense grid of 50 points does not crash the Cholesky factor, and the same seed repeats")
def test_prior_jitter(sub):
    fn = has(sub, "gp_prior_samples")
    X = np.linspace(0.0, 6.0, 50)[:, None]
    kern = lambda a, b: ref_rbf(a, b, 0.7)  # noqa: E731
    try:
        a = fn(X, kern, 3, np.random.default_rng(1))
    except np.linalg.LinAlgError:
        fail("the Cholesky factor failed on a 50-point grid; add jitter 1e-8 * I to K before factoring")
        return
    shape(a, (3, 50), label="samples")
    finite(a, label="samples")
    b = fn(X, kern, 3, np.random.default_rng(1))
    close(a, b, atol=0.0, label="two runs with the same seed")


@check("gp_posterior: matches the two-point hand example, mean 1.0157 and variance 0.4494 at a training point")
def test_posterior_hand(sub):
    fn = has(sub, "gp_posterior")
    X, y, kern = two_points()
    mean, var = fn(X, y, np.array([[0.0], [0.5]]), kern, 1.0)
    shape(mean, (2,), label="mean")
    shape(var, (2,), label="variance")
    close(mean, np.array([0.78333905, 1.01571439]), atol=1e-6, label="posterior mean at x = 0 and 0.5")
    close(var, np.array([0.44935748, 0.40242346]), atol=1e-6, label="posterior variance at x = 0 and 0.5")


@check("gp_posterior: the mean equals the kernel ridge prediction with lambda equal to the noise variance")
def test_posterior_mean_is_kernel_ridge(sub):
    fn = has(sub, "gp_posterior")
    rng = np.random.default_rng(4)
    X = np.sort(rng.uniform(0, 6, size=12))[:, None]
    y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=12)
    Xs = np.linspace(-1, 7, 30)[:, None]
    kern = lambda a, b: ref_rbf(a, b, 0.9)  # noqa: E731
    noise = 0.15
    alpha = np.linalg.solve(kern(X, X) + noise * np.eye(12), y)
    want = kern(Xs, X) @ alpha
    mean, _ = fn(X, y, Xs, kern, noise)
    close(mean, want, atol=1e-8, rtol=1e-7, label="posterior mean against kernel ridge")


@check("gp_posterior: the variance is non-negative, about noise / (1 + noise) at training points, and the prior variance far away")
def test_posterior_variance(sub):
    fn = has(sub, "gp_posterior")
    # Training points 10 apart with length scale 1 have K close to the identity.
    X = np.array([[0.0], [10.0], [20.0]])
    y = np.array([0.5, -1.0, 2.0])
    kern = lambda a, b: ref_rbf(a, b, 1.0)  # noqa: E731
    noise = 0.1
    _, var = fn(X, y, X, kern, noise)
    close(var, np.full(3, noise / (1.0 + noise)), atol=1e-6, label="variance at the training points")
    if np.any(var > noise + 1e-9):
        fail(f"variance at a training point is {var}, which is above the noise level {noise}")
    _, far = fn(X, y, np.array([[100.0], [-50.0]]), kern, noise)
    close(far, np.ones(2), atol=1e-6, label="variance far from the data")
    rng = np.random.default_rng(5)
    X2 = np.sort(rng.uniform(0, 6, size=15))[:, None]
    y2 = rng.normal(size=15)
    grid = np.linspace(-2, 8, 200)[:, None]
    _, v2 = fn(X2, y2, grid, lambda a, b: ref_rbf(a, b, 0.8), 0.05)
    finite(v2, label="variance on a grid")
    if v2.min() < -1e-9:
        fail(f"variance went negative: smallest value {v2.min():.3g}")
    if v2.max() > 1.0 + 1e-9:
        fail(f"variance {v2.max():.6g} is above the prior variance 1")


@check("gp_posterior: with tiny noise the mean interpolates the data and the variance at the data collapses")
def test_posterior_interpolates(sub):
    fn = has(sub, "gp_posterior")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.3, -0.7, 1.1, 0.2])
    kern = lambda a, b: ref_rbf(a, b, 1.0)  # noqa: E731
    mean, var = fn(X, y, X, kern, 1e-6)
    close(mean, y, atol=1e-3, label="posterior mean at the training inputs")
    if np.any(var > 1e-5) or np.any(var < -1e-9):
        fail(f"variance at the training inputs should be near 0 for tiny noise, got {var}")


@check("log_marginal_likelihood: matches the two-point hand value -3.5254 and an independent Gaussian density")
def test_lml_value(sub):
    fn = has(sub, "log_marginal_likelihood")
    X, y, _ = two_points()
    almost(fn(X, y, 1.0, 1.0), -3.5254101132, 1e-8, label="two-point log marginal likelihood")
    rng = np.random.default_rng(6)
    X2 = rng.uniform(0, 5, size=(9, 2))
    y2 = rng.normal(size=9)
    for ls, noise in ((0.6, 0.05), (1.7, 0.3)):
        cov = ref_rbf(X2, X2, ls) + noise * np.eye(9)
        want = multivariate_normal(mean=np.zeros(9), cov=cov).logpdf(y2)
        almost(fn(X2, y2, ls, noise), want, 1e-8, label=f"length scale {ls}, noise {noise}")


@check("lml_grad_log_length_scale: matches a finite difference in log(length scale) at two length scales")
def test_lml_gradient(sub):
    fn = has(sub, "lml_grad_log_length_scale")
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 4, size=(10, 2))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=10)
    noise = 0.1
    for ls in (0.7, 2.0):
        want = finite_difference(lambda t: ref_lml(X, y, float(np.exp(t[0])), noise), np.array([np.log(ls)]))[0]
        almost(fn(X, y, ls, noise), want, 1e-5, label=f"slope in log length scale at {ls}")


@check("best_length_scale: on data drawn with length scale 1, the evidence picks a value close to 1")
def test_best_length_scale(sub):
    fn = has(sub, "best_length_scale")
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(0, 8, size=40))[:, None]
    K = ref_rbf(X, X, 1.0) + 1e-8 * np.eye(40)
    f = np.linalg.cholesky(K) @ rng.normal(size=40)
    y = f + 0.2 * rng.normal(size=40)
    noise = 0.04
    grid = np.geomspace(0.1, 10.0, 41)
    best = fn(X, y, noise, grid)
    if not (1.0 / 1.5 <= best <= 1.5):
        fail(f"best length scale is {best:.3g}, expected a value within a factor of 1.5 of the true 1.0")
    if best not in set(float(g) for g in grid):
        fail(f"best length scale {best} is not an entry of the grid")
    lml_true = ref_lml(X, y, 1.0, noise)
    if not (lml_true > ref_lml(X, y, 0.1, noise) and lml_true > ref_lml(X, y, 10.0, noise)):
        fail("test data problem: the evidence at 1.0 should beat 0.1 and 10")
