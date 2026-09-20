"""Generative classifiers: Gaussian discriminant analysis and naive Bayes.

Fill in the five functions below. `fit_gda` and `gda_log_posterior` build a
shared-covariance Gaussian discriminant analysis (GDA) classifier.
`gda_to_logistic` extracts the logistic regression that GDA's posterior is
secretly equal to. `fit_naive_bayes_gaussian` and `fit_naive_bayes_multinomial`
build the conditional-independence special case for continuous and count
features.

Labels `y` are integers `0, ..., K - 1`. Nothing here needs iterative
optimisation: every quantity has a closed-form maximum likelihood estimate.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def fit_gda(X: np.ndarray, y: np.ndarray):
    """Fit Gaussian discriminant analysis with a covariance shared across classes.

    Args:
        X: Features, shape (n, d).
        y: Integer class labels in {0, ..., K - 1}, shape (n,).

    Returns:
        A tuple `(means, cov, priors)`:
        means: Per-class mean, shape (K, d).
        cov: The pooled covariance shared by every class, shape (d, d), the
            maximum likelihood estimate (divide by n, not n - K).
        priors: Per-class prior p(y = k), shape (K,), summing to 1.
    """
    raise NotImplementedError("fit_gda")


def gda_log_posterior(
    X: np.ndarray, means: np.ndarray, cov: np.ndarray, priors: np.ndarray
) -> np.ndarray:
    """Return the log posterior log p(y = k | x) under the GDA model.

    Use a Cholesky factor of `cov` to evaluate the Gaussian log-density
    without forming its inverse or determinant directly, then normalise with
    a log-sum-exp so each row is a valid log-probability distribution.

    Args:
        X: Features, shape (n, d).
        means: Per-class mean, shape (K, d).
        cov: Shared covariance, shape (d, d).
        priors: Per-class prior, shape (K,).

    Returns:
        Log posterior probabilities, shape (n, K). `np.exp` of each row sums
        to 1.
    """
    raise NotImplementedError("gda_log_posterior")


def gda_to_logistic(means: np.ndarray, cov: np.ndarray, priors: np.ndarray):
    """Convert a two-class GDA model to the logistic regression it induces.

    With a covariance shared between the two classes, the posterior
    p(y = 1 | x) is exactly `sigmoid(w @ x + b)` for a specific `w` and `b`.
    Derive and return them.

    Args:
        means: Per-class mean, shape (2, d).
        cov: Shared covariance, shape (d, d).
        priors: Per-class prior, shape (2,).

    Returns:
        A tuple `(w, b)`: `w` has shape (d,), `b` is a float.
    """
    raise NotImplementedError("gda_to_logistic")


def fit_naive_bayes_gaussian(X: np.ndarray, y: np.ndarray, var_floor: float = 1e-9):
    """Fit a Gaussian naive Bayes classifier: one univariate Gaussian per feature per class.

    Unlike `fit_gda`, nothing is shared across classes or across features:
    each class gets its own diagonal covariance, stored as a variance vector.

    Args:
        X: Features, shape (n, d).
        y: Integer class labels in {0, ..., K - 1}, shape (n,).
        var_floor: Minimum variance, to avoid dividing by zero on a feature
            that is constant within a class.

    Returns:
        A tuple `(means, variances, priors)`, each of shape (K, d), (K, d),
        and (K,) respectively.
    """
    raise NotImplementedError("fit_naive_bayes_gaussian")


def fit_naive_bayes_multinomial(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Fit a multinomial naive Bayes classifier over count features, with Laplace smoothing.

    Args:
        X: Nonnegative count features (for example, word counts), shape (n, V).
        y: Integer class labels in {0, ..., K - 1}, shape (n,).
        alpha: The Laplace smoothing pseudocount added to every entry.

    Returns:
        A tuple `(log_theta, log_priors)`:
        log_theta: log p(word j | class k), shape (K, V). `np.exp` of each row
            sums to 1.
        log_priors: log p(y = k), shape (K,).
    """
    raise NotImplementedError("fit_naive_bayes_multinomial")


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
