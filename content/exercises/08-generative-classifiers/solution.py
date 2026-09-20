"""Reference solution for the generative classifiers exercise."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def _log_sum_exp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def fit_gda(X: np.ndarray, y: np.ndarray):
    n, d = X.shape
    K = int(y.max()) + 1
    means = np.zeros((K, d))
    cov = np.zeros((d, d))
    priors = np.zeros(K)
    for k in range(K):
        mask = y == k
        nk = int(mask.sum())
        priors[k] = nk / n
        means[k] = X[mask].mean(axis=0)
        diff = X[mask] - means[k]
        cov += diff.T @ diff
    cov /= n
    return means, cov, priors


def gda_log_posterior(
    X: np.ndarray, means: np.ndarray, cov: np.ndarray, priors: np.ndarray
) -> np.ndarray:
    n, d = X.shape
    K = means.shape[0]
    L = cholesky(cov, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    joint = np.empty((n, K))
    for k in range(K):
        diff = (X - means[k]).T
        z = solve_triangular(L, diff, lower=True)
        mahalanobis = np.sum(z**2, axis=0)
        log_likelihood = -0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det - 0.5 * mahalanobis
        joint[:, k] = np.log(priors[k]) + log_likelihood
    return joint - _log_sum_exp(joint, axis=1)[:, None]


def gda_to_logistic(means: np.ndarray, cov: np.ndarray, priors: np.ndarray):
    if means.shape[0] != 2:
        raise ValueError("gda_to_logistic only handles two classes")
    mu0, mu1 = means[0], means[1]
    precision = np.linalg.inv(cov)
    w = precision @ (mu1 - mu0)
    b = (
        np.log(priors[1] / priors[0])
        - 0.5 * mu1 @ precision @ mu1
        + 0.5 * mu0 @ precision @ mu0
    )
    return w, float(b)


def fit_naive_bayes_gaussian(X: np.ndarray, y: np.ndarray, var_floor: float = 1e-9):
    n, d = X.shape
    K = int(y.max()) + 1
    means = np.zeros((K, d))
    variances = np.zeros((K, d))
    priors = np.zeros(K)
    for k in range(K):
        mask = y == k
        nk = int(mask.sum())
        priors[k] = nk / n
        means[k] = X[mask].mean(axis=0)
        variances[k] = np.maximum(X[mask].var(axis=0), var_floor)
    return means, variances, priors


def fit_naive_bayes_multinomial(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    n, V = X.shape
    K = int(y.max()) + 1
    log_theta = np.empty((K, V))
    log_priors = np.empty(K)
    for k in range(K):
        mask = y == k
        log_priors[k] = np.log(mask.sum() / n)
        counts = X[mask].sum(axis=0)
        smoothed = counts + alpha
        log_theta[k] = np.log(smoothed / smoothed.sum())
    return log_theta, log_priors


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 3
    mu0, mu1 = np.zeros(d), np.array([2.0, 0.0, 0.0])
    n0, n1 = 60, 60
    X = np.vstack([mu0 + rng.normal(size=(n0, d)), mu1 + rng.normal(size=(n1, d))])
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)

    means, cov, priors = fit_gda(X, y)
    print("GDA means:\n", means)
    print("GDA pooled covariance:\n", cov)

    log_post = gda_log_posterior(X, means, cov, priors)
    accuracy = (log_post.argmax(axis=1) == y).mean()
    print(f"training accuracy: {accuracy:.3f}")

    w, b = gda_to_logistic(means, cov, priors)
    print(f"equivalent logistic weights: w={w}, b={b:.3f}")

    nb_means, nb_vars, nb_priors = fit_naive_bayes_gaussian(X, y)
    print("Gaussian naive Bayes per-class variances:\n", nb_vars)

    vocab = 6
    docs = rng.integers(0, 4, size=(20, vocab))
    labels = rng.integers(0, 2, size=20)
    log_theta, log_priors = fit_naive_bayes_multinomial(docs, labels, alpha=1.0)
    print("multinomial row sums:", np.exp(log_theta).sum(axis=1))
