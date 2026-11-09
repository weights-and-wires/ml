"""Checks for likelihood, MAP, and Bayesian linear regression.

Three plausible wrong implementations are named below, and each check that
would accept one of them fails it with the expected value and the value it
returns:

- map_objective_without_prior returns only the Gaussian negative log likelihood
  and drops the prior term.
- posterior_linreg_swapped_variances uses tau2 * (X.T @ X + (tau2/sigma2) I)^-1
  in place of sigma2 * (X.T @ X + (sigma2/tau2) I)^-1.
- predictive_forgets_noise returns x.T @ Sigma @ x and drops the noise variance.
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


def houses():
    x_mat = np.array([[1.0, 0.0], [1.0, 2.0]])
    y = np.array([0.0, 2.0])
    return x_mat, y


def map_objective_without_prior(X, y, w, sigma2, tau2):
    """Wrong: the prior term is missing, so the score is only the NLL."""
    del tau2
    resid = X @ np.asarray(w, dtype=float) - y
    n = X.shape[0]
    return float(0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * np.dot(resid, resid) / sigma2)


def posterior_linreg_swapped_variances(X, y, sigma2, tau2):
    """Wrong: sigma2 and tau2 are swapped in the fee and in the scale."""
    del y
    d = X.shape[1]
    gram = X.T @ X + (tau2 / sigma2) * np.eye(d)
    return tau2 * np.linalg.solve(gram, np.eye(d))


def predictive_forgets_noise(x_new, cov):
    """Wrong: the predictive variance is only x.T @ Sigma @ x."""
    row = np.asarray(x_new, dtype=float)
    return float(row @ cov @ row)


@check("neg_log_likelihood_gaussian matches the two houses: log(2 pi) at the perfect fit, and log(2 pi) + 10/121 at the MAP weights")
def test_gaussian_nll(sub):
    fn = has(sub, "neg_log_likelihood_gaussian")
    x_mat, y = houses()
    perfect = np.array([0.0, 1.0])
    almost(
        fn(x_mat, y, perfect, 1.0),
        np.log(2 * np.pi),
        1e-12,
        label="NLL at w = (0, 1), sigma2 = 1",
    )
    shrunk = np.array([2.0 / 11.0, 8.0 / 11.0])
    almost(
        fn(x_mat, y, shrunk, 1.0),
        np.log(2 * np.pi) + 10.0 / 121.0,
        1e-12,
        label="NLL at w = (2/11, 8/11), sigma2 = 1",
    )
    # sigma2 under the squared misses: 20/121 squared length, divided by 2 * 0.25.
    almost(
        fn(x_mat, y, shrunk, 0.25),
        np.log(2 * np.pi * 0.25) + 40.0 / 121.0,
        1e-12,
        label="NLL at the same weights with sigma2 = 0.25",
    )


@check("map_objective adds the Gaussian prior, and map_objective_without_prior does not match that score")
def test_map_objective(sub):
    fn = has(sub, "map_objective")
    x_mat, y = houses()
    shrunk = np.array([2.0 / 11.0, 8.0 / 11.0])
    want = 2 * np.log(2 * np.pi) + 44.0 / 121.0
    almost(fn(x_mat, y, shrunk, 1.0, 1.0), want, 1e-12, label="MAP score at (2/11, 8/11)")
    # n = 3 rows and d = 1 weight, so a prior log that uses n instead of d fails.
    design = np.array([[1.0], [0.0], [2.0]])
    labels = np.array([1.0, 0.5, -1.0])
    weight = np.array([0.4])
    sigma2, tau2 = 0.5, 2.0
    resid = design @ weight - labels
    nll = 0.5 * 3 * np.log(2 * np.pi * sigma2) + 0.5 * np.dot(resid, resid) / sigma2
    prior = 0.5 * np.log(2 * np.pi * tau2) + 0.5 * np.dot(weight, weight) / tau2
    almost(
        fn(design, labels, weight, sigma2, tau2),
        nll + prior,
        1e-10,
        label="MAP score with 3 rows and 1 weight",
    )
    missing = map_objective_without_prior(x_mat, y, shrunk, 1.0, 1.0)
    if abs(missing - want) <= 1e-8:
        fail(
            f"map_objective_without_prior returned {missing:.6g}, expected a gap "
            f"from the full MAP score {want:.6g}"
        )


@check("map_gradient matches a finite difference of map_objective")
def test_map_gradient(sub):
    grad_fn = has(sub, "map_gradient")
    objective = has(sub, "map_objective")
    rng = np.random.default_rng(0)
    design = rng.normal(size=(18, 4))
    labels = rng.normal(size=18)
    for sigma2, tau2 in ((1.0, 1.0), (0.3, 2.5), (4.0, 0.5)):
        weights = rng.normal(size=4)
        got = grad_fn(design, labels, weights, sigma2, tau2)
        shape(got, (4,), label="map_gradient output")
        want = finite_difference(
            lambda v, s2=sigma2, t2=tau2: objective(design, labels, v, s2, t2),
            weights.copy(),
        )
        close(got, want, rtol=1e-5, atol=1e-5, label=f"gradient at sigma2 = {sigma2}, tau2 = {tau2}")


@check("ridge_weights(lambda = sigma2/tau2) is where a finite difference of map_objective is 0")
def test_ridge_is_map(sub):
    ridge = has(sub, "ridge_weights")
    objective = has(sub, "map_objective")
    x_mat, y = houses()
    got = ridge(x_mat, y, 1.0)
    shape(got, (2,), label="ridge_weights output")
    close(got, np.array([2.0 / 11.0, 8.0 / 11.0]), atol=1e-10, label="weights on the two houses at lambda = 1")
    grad = finite_difference(lambda v: objective(x_mat, y, v, 1.0, 1.0), got.copy())
    close(grad, np.zeros(2), atol=1e-6, label="finite-difference gradient of map_objective at the ridge weights")
    perfect = np.array([0.0, 1.0])
    leftover = finite_difference(lambda v: objective(x_mat, y, v, 1.0, 1.0), perfect.copy())
    if np.linalg.norm(leftover) < 1e-4:
        fail(
            f"finite-difference gradient at the least-squares weights is {leftover}, "
            "expected a nonzero prior pull"
        )
    rng = np.random.default_rng(1)
    design = rng.normal(size=(25, 3))
    labels = design @ np.array([0.5, -1.0, 0.25]) + rng.normal(size=25)
    sigma2, tau2 = 0.4, 1.5
    fitted = ridge(design, labels, sigma2 / tau2)
    moved = finite_difference(
        lambda v: objective(design, labels, v, sigma2, tau2),
        fitted.copy(),
    )
    close(moved, np.zeros(3), atol=1e-5, label="finite-difference gradient at ridge_weights for sigma2 = 0.4, tau2 = 1.5")


@check("posterior_linreg mean matches ridge_weights at lambda = sigma2/tau2")
def test_posterior_mean(sub):
    posterior = has(sub, "posterior_linreg")
    ridge = has(sub, "ridge_weights")
    x_mat, y = houses()
    mean, cov = posterior(x_mat, y, 1.0, 1.0)
    shape(mean, (2,), label="posterior mean")
    shape(cov, (2, 2), label="posterior covariance")
    close(mean, ridge(x_mat, y, 1.0), atol=1e-10, label="posterior mean against ridge_weights on the two houses")
    rng = np.random.default_rng(2)
    design = rng.normal(size=(30, 4))
    labels = rng.normal(size=30)
    sigma2, tau2 = 0.25, 4.0
    mean, _cov = posterior(design, labels, sigma2, tau2)
    close(
        mean,
        ridge(design, labels, sigma2 / tau2),
        atol=1e-8,
        label="posterior mean against ridge_weights at sigma2 = 0.25, tau2 = 4",
    )


@check("posterior covariance is sigma2 (X.T X + (sigma2/tau2) I)^-1, and the swapped-variance covariance is not")
def test_posterior_covariance(sub):
    posterior = has(sub, "posterior_linreg")
    x_mat, y = houses()
    _mean, cov = posterior(x_mat, y, 1.0, 1.0)
    want = np.array([[5.0, -2.0], [-2.0, 3.0]]) / 11.0
    close(cov, want, atol=1e-10, label="covariance on the two houses")
    rng = np.random.default_rng(3)
    design = rng.normal(size=(20, 3))
    labels = rng.normal(size=20)
    sigma2, tau2 = 0.25, 4.0
    _mean, cov = posterior(design, labels, sigma2, tau2)
    lam = sigma2 / tau2
    gram = design.T @ design + lam * np.eye(3)
    want = sigma2 * np.linalg.solve(gram, np.eye(3))
    swapped = posterior_linreg_swapped_variances(design, labels, sigma2, tau2)
    if np.allclose(swapped, want, atol=1e-8):
        fail(
            f"posterior_linreg_swapped_variances matched the true covariance.\n"
            f"  got:  {swapped}\n"
            f"  want: {want}"
        )
    if np.allclose(cov, swapped, atol=1e-6):
        fail(
            f"covariance swaps sigma2 and tau2.\n"
            f"  got:  {cov}\n"
            f"  want: {want}"
        )
    close(cov, want, atol=1e-8, label="covariance at sigma2 = 0.25, tau2 = 4")


@check("the posterior covariance shrinks when the same houses are observed again")
def test_posterior_shrinks(sub):
    posterior = has(sub, "posterior_linreg")
    x_mat, y = houses()
    _mean, once = posterior(x_mat, y, 1.0, 1.0)
    twice_x = np.vstack([x_mat, x_mat])
    twice_y = np.concatenate([y, y])
    _mean, twice = posterior(twice_x, twice_y, 1.0, 1.0)
    trace_once = float(np.trace(once))
    trace_twice = float(np.trace(twice))
    almost(trace_once, 8.0 / 11.0, 1e-10, label="trace after two houses")
    almost(trace_twice, 14.0 / 29.0, 1e-10, label="trace after those houses again")
    if trace_twice >= trace_once:
        fail(
            f"trace after a second look is {trace_twice:.6g}, expected it below "
            f"the first trace {trace_once:.6g}"
        )
    rng = np.random.default_rng(4)
    design = rng.normal(size=(12, 3))
    labels = rng.normal(size=12)
    extra = rng.normal(size=(24, 3))
    _mean, small = posterior(design, labels, 1.0, 1.0)
    _mean, large = posterior(
        np.vstack([design, extra]),
        np.concatenate([labels, rng.normal(size=24)]),
        1.0,
        1.0,
    )
    if float(np.trace(large)) >= float(np.trace(small)):
        fail(
            f"trace with 36 rows is {float(np.trace(large)):.6g}, expected it below "
            f"the trace {float(np.trace(small)):.6g} from the first 12 rows"
        )


@check("predictive variance is sigma2 + x.T Sigma x, larger at size 6 than at the studio, and predictive_forgets_noise fails")
def test_predictive(sub):
    posterior = has(sub, "posterior_linreg")
    predict = has(sub, "predictive")
    x_mat, y = houses()
    mean, cov = posterior(x_mat, y, 1.0, 1.0)
    studio = np.array([1.0, 0.0])
    far = np.array([1.0, 6.0])
    mean0, var0 = predict(studio, mean, cov, 1.0)
    mean6, var6 = predict(far, mean, cov, 1.0)
    almost(mean0, 2.0 / 11.0, 1e-10, label="predictive mean at size 0")
    almost(mean6, 50.0 / 11.0, 1e-10, label="predictive mean at size 6")
    forgotten = predictive_forgets_noise(studio, np.array([[5.0, -2.0], [-2.0, 3.0]]) / 11.0)
    if abs(forgotten - 16.0 / 11.0) <= 1e-8:
        fail(
            f"predictive_forgets_noise returned {forgotten:.6g}, expected it to "
            f"differ from 16/11"
        )
    if abs(float(var0) - forgotten) <= 1e-6:
        fail(
            f"predictive variance at the studio is {float(var0):.6g}, which is "
            f"x.T Sigma x and forgets the noise. Expected {16.0 / 11.0:.6g}."
        )
    almost(var0, 16.0 / 11.0, 1e-10, label="predictive variance at size 0")
    almost(var6, 100.0 / 11.0, 1e-10, label="predictive variance at size 6")
    if float(var6) <= float(var0):
        fail(
            f"variance at size 6 is {float(var6):.6g}, expected it above the studio "
            f"variance {float(var0):.6g}"
        )
