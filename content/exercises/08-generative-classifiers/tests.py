"""Checks for the generative classifiers exercise."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from mlcheck import check, close, fail, has, shape


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _fit_logistic_reference(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, maxiter: int = 500):
    """A discriminative baseline, independent of the reader's code, used only to
    demonstrate the win/loss pattern against GDA. It fits by minimising the
    (ridge-penalised) negative log-likelihood directly."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])

    def nll(w):
        p = _sigmoid(Xb @ w)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return loss + 0.5 * l2 * np.sum(w[:-1] ** 2) / n

    def grad(w):
        p = _sigmoid(Xb @ w)
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        return g

    result = minimize(nll, np.zeros(d + 1), jac=grad, method="L-BFGS-B", options={"maxiter": maxiter})
    return result.x[:-1], result.x[-1]


@check("fit_gda returns correctly shaped output with priors that sum to one")
def test_fit_gda_shapes(sub):
    got = has(sub, "fit_gda")
    rng = np.random.default_rng(0)
    d, K = 3, 3
    counts = [20, 15, 25]
    X = np.vstack([rng.normal(loc=k * 2.0, size=(counts[k], d)) for k in range(K)])
    y = np.concatenate([np.full(counts[k], k) for k in range(K)])

    means, cov, priors = got(X, y)
    shape(means, (K, d), label="means")
    shape(cov, (d, d), label="covariance")
    shape(priors, (K,), label="priors")
    total = float(np.sum(priors))
    if abs(total - 1.0) > 1e-10:
        fail(f"priors must sum to 1, but they sum to {total:.6f}")


@check("fit_gda's means, covariance, and priors match a direct computation of the MLE")
def test_fit_gda_mle(sub):
    got = has(sub, "fit_gda")
    rng = np.random.default_rng(1)
    d, K = 2, 3
    counts = [12, 18, 10]
    n = sum(counts)
    X = np.vstack([rng.normal(loc=[k * 3.0, -k], scale=1.5, size=(counts[k], d)) for k in range(K)])
    y = np.concatenate([np.full(counts[k], k) for k in range(K)])

    means, cov, priors = got(X, y)

    want_means = np.array([X[y == k].mean(axis=0) for k in range(K)])
    close(means, want_means, atol=1e-10, label="class means")

    want_priors = np.array(counts) / n
    close(priors, want_priors, atol=1e-10, label="class priors")

    want_cov = np.zeros((d, d))
    for k in range(K):
        diff = X[y == k] - want_means[k]
        want_cov += diff.T @ diff
    want_cov /= n
    close(cov, want_cov, atol=1e-10, rtol=1e-10, label="pooled covariance")


@check("gda_log_posterior exponentiates to a valid probability distribution")
def test_log_posterior_normalises(sub):
    got = has(sub, "gda_log_posterior")
    rng = np.random.default_rng(2)
    d, K = 4, 3
    means = rng.normal(size=(K, d)) * 2.0
    A = rng.normal(size=(d, d))
    cov = A @ A.T + d * np.eye(d)
    priors = np.array([0.2, 0.5, 0.3])
    X = rng.normal(size=(30, d))

    log_post = got(X, means, cov, priors)
    shape(log_post, (30, K), label="log posterior")
    row_sums = np.exp(log_post).sum(axis=1)
    close(row_sums, np.ones(30), atol=1e-8, label="exponentiated rows")


@check("gda_log_posterior matches an independent Gaussian log-density computation")
def test_log_posterior_matches_reference(sub):
    got = has(sub, "gda_log_posterior")
    rng = np.random.default_rng(3)
    d, K = 3, 2
    means = np.array([[0.0, 0.0, 0.0], [1.5, -1.0, 0.5]])
    A = rng.normal(size=(d, d))
    cov = A @ A.T + d * np.eye(d)
    priors = np.array([0.4, 0.6])
    X = rng.normal(size=(25, d)) + 0.5

    joint = np.column_stack(
        [np.log(priors[k]) + multivariate_normal(means[k], cov).logpdf(X) for k in range(K)]
    )
    m = joint.max(axis=1, keepdims=True)
    want = joint - (m + np.log(np.exp(joint - m).sum(axis=1, keepdims=True)))

    log_post = got(X, means, cov, priors)
    close(log_post, want, atol=1e-8, rtol=1e-6, label="log posterior")


@check("gda_to_logistic reproduces GDA's decision function to machine precision")
def test_gda_to_logistic(sub):
    to_logistic = has(sub, "gda_to_logistic")
    log_posterior = has(sub, "gda_log_posterior")
    rng = np.random.default_rng(4)
    d = 5
    means = rng.normal(size=(2, d)) * 1.5
    A = rng.normal(size=(d, d))
    cov = A @ A.T + d * np.eye(d)
    priors = np.array([0.35, 0.65])
    X = rng.normal(size=(40, d))

    w, b = to_logistic(means, cov, priors)
    shape(np.asarray(w), (d,), label="w")

    from_logistic = _sigmoid(X @ np.asarray(w) + float(b))
    from_gda = np.exp(log_posterior(X, means, cov, priors)[:, 1])
    close(from_logistic, from_gda, atol=1e-8, rtol=1e-6, label="p(y=1 | x)")


@check("fit_naive_bayes_gaussian matches the per-class, per-feature MLE")
def test_naive_bayes_gaussian(sub):
    got = has(sub, "fit_naive_bayes_gaussian")
    rng = np.random.default_rng(5)
    d, K = 4, 2
    counts = [30, 40]
    X = np.vstack(
        [rng.normal(loc=k * 1.5, scale=[1.0, 2.0, 0.5, 3.0], size=(counts[k], d)) for k in range(K)]
    )
    y = np.concatenate([np.full(counts[k], k) for k in range(K)])

    means, variances, priors = got(X, y)
    shape(means, (K, d), label="means")
    shape(variances, (K, d), label="variances")

    for k in range(K):
        close(means[k], X[y == k].mean(axis=0), atol=1e-10, label=f"class {k} means")
        close(variances[k], X[y == k].var(axis=0), atol=1e-10, label=f"class {k} variances")
    close(priors, np.array(counts) / sum(counts), atol=1e-10, label="priors")


@check("fit_naive_bayes_multinomial applies Laplace smoothing correctly and normalises")
def test_naive_bayes_multinomial(sub):
    got = has(sub, "fit_naive_bayes_multinomial")
    rng = np.random.default_rng(6)
    V, K = 6, 2
    counts = [8, 5]
    docs = [rng.integers(0, 4, size=(counts[k], V)) for k in range(K)]
    X = np.vstack(docs)
    y = np.concatenate([np.full(counts[k], k) for k in range(K)])
    alpha = 1.0

    log_theta, log_priors = got(X, y, alpha)
    shape(log_theta, (K, V), label="log_theta")
    shape(log_priors, (K,), label="log_priors")

    row_sums = np.exp(log_theta).sum(axis=1)
    close(row_sums, np.ones(K), atol=1e-8, label="exponentiated rows")

    for k in range(K):
        raw_counts = X[y == k].sum(axis=0)
        want = np.log((raw_counts + alpha) / (raw_counts.sum() + alpha * V))
        close(log_theta[k], want, atol=1e-10, label=f"class {k} smoothed log-probabilities")

    want_log_priors = np.log(np.array(counts) / sum(counts))
    close(log_priors, want_log_priors, atol=1e-10, label="log priors")


@check("GDA beats logistic regression when the Gaussian assumption holds")
def test_gda_beats_logistic_when_correct(sub):
    fit_gda = has(sub, "fit_gda")
    log_posterior = has(sub, "gda_log_posterior")
    d = 3
    mu0, mu1 = np.zeros(d), np.array([2.0, 0.0, 0.0])

    gda_accuracy, logistic_accuracy = [], []
    for trial in range(1500):
        rng = np.random.default_rng(5000 + trial)
        n0 = n1 = 6
        X_train = np.vstack([mu0 + rng.normal(size=(n0, d)), mu1 + rng.normal(size=(n1, d))])
        y_train = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)

        n_test = 1000
        X_test = np.vstack(
            [mu0 + rng.normal(size=(n_test // 2, d)), mu1 + rng.normal(size=(n_test // 2, d))]
        )
        y_test = np.concatenate([np.zeros(n_test // 2), np.ones(n_test // 2)]).astype(int)

        means, cov, priors = fit_gda(X_train, y_train)
        gda_pred = log_posterior(X_test, means, cov, priors).argmax(axis=1)
        gda_accuracy.append((gda_pred == y_test).mean())

        w, b = _fit_logistic_reference(X_train, y_train)
        logistic_pred = (_sigmoid(X_test @ w + b) >= 0.5).astype(int)
        logistic_accuracy.append((logistic_pred == y_test).mean())

    gap = float(np.mean(gda_accuracy) - np.mean(logistic_accuracy))
    if gap <= 0.002:
        fail(
            "averaged over 1500 small training sets drawn from two Gaussians with a "
            "shared covariance, GDA should beat a directly fit logistic regression "
            f"by a small but consistent margin, but the observed gap is {gap:.4f}"
        )


@check("GDA loses to logistic regression once the Gaussian assumption is broken by outliers")
def test_gda_loses_when_wrong(sub):
    fit_gda = has(sub, "fit_gda")
    log_posterior = has(sub, "gda_log_posterior")
    d = 3
    mu0, mu1 = np.zeros(d), np.array([2.0, 0.0, 0.0])

    gda_accuracy, logistic_accuracy = [], []
    for trial in range(400):
        rng = np.random.default_rng(9000 + trial)
        n0 = n1 = 30
        X0 = mu0 + rng.normal(size=(n0, d))
        X1 = mu1 + rng.normal(size=(n1, d))
        # Three points, correctly classified but far along a random direction:
        # they barely change the Bayes-optimal boundary, but they drag the
        # class-1 mean and covariance a long way from their true values.
        n_out = 3
        direction = rng.normal(size=(n_out, d))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        X1 = np.vstack([X1, mu1 + 20.0 * direction])
        X_train = np.vstack([X0, X1])
        y_train = np.concatenate([np.zeros(n0), np.ones(n1 + n_out)]).astype(int)

        n_test = 1000
        X_test = np.vstack(
            [mu0 + rng.normal(size=(n_test // 2, d)), mu1 + rng.normal(size=(n_test // 2, d))]
        )
        y_test = np.concatenate([np.zeros(n_test // 2), np.ones(n_test // 2)]).astype(int)

        means, cov, priors = fit_gda(X_train, y_train)
        gda_pred = log_posterior(X_test, means, cov, priors).argmax(axis=1)
        gda_accuracy.append((gda_pred == y_test).mean())

        w, b = _fit_logistic_reference(X_train, y_train)
        logistic_pred = (_sigmoid(X_test @ w + b) >= 0.5).astype(int)
        logistic_accuracy.append((logistic_pred == y_test).mean())

    gap = float(np.mean(gda_accuracy) - np.mean(logistic_accuracy))
    if gap >= -0.004:
        fail(
            "averaged over 400 trials where three outliers contaminate the "
            "training data without touching the Bayes-optimal boundary, GDA's "
            "mean and covariance estimates should be dragged off enough that a "
            f"directly fit logistic regression wins, but the observed gap is {gap:.4f}"
        )
