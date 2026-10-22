"""The E-step of a Gaussian mixture.

Fill in the three functions below, in order. Each one is built from the one
before it.
"""

import numpy as np


def log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of one Gaussian at every row of X.

    Args:
        X: Points, shape (n, d).
        mean: The blob's centre, shape (d,).
        cov: The blob's covariance, shape (d, d), positive definite.

    Returns:
        Shape (n,). Entry i is
        -0.5 * (d * log(2 pi) + log det(cov) + (x_i - mean)^T cov^-1 (x_i - mean)).

    Use np.linalg.slogdet for the log determinant, which stays finite when the
    determinant itself would underflow.
    """
    raise NotImplementedError("log_gaussian")


def logsumexp_rows(s: np.ndarray) -> np.ndarray:
    """Return log(sum_j exp(s[i, j])) for every row, without underflow.

    Args:
        s: Scores, shape (n, k). They can be as low as -1000 or lower.

    Returns:
        Shape (n,). Subtract each row's maximum before exponentiating, and add it
        back after the log. Do not modify s.
    """
    raise NotImplementedError("logsumexp_rows")


def gmm_e_step(X: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray):
    """Compute each blob's responsibility for each point, and the log-likelihood.

    Args:
        X: Points, shape (n, d).
        means: Blob centres, shape (k, d).
        covs: Blob covariances, shape (k, d, d).
        weights: Blob weights, shape (k,), adding up to 1.

    Returns:
        A tuple (responsibilities, log_likelihood):
            responsibilities: shape (n, k). Entry (i, j) is the probability that
                blob j produced point i, so every row adds up to 1.
            log_likelihood: the sum over points of the log of the weighted
                density, as a Python float.
    """
    raise NotImplementedError("gmm_e_step")


if __name__ == "__main__":
    # The chapter's two points, and two blobs with means 0 and 2.
    X = np.array([[0.0], [2.0]])
    means = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])

    print("log density at 0:", log_gaussian(X, means[0], covs[0]))  # [-0.9189 -2.9189]
    print("log-sum-exp of -1000, -1001:", logsumexp_rows(np.array([[-1000.0, -1001.0]])))  # about [-999.6867]
    resp, log_l = gmm_e_step(X, means, covs, weights)
    print("responsibilities:\n", np.round(resp, 6))  # [[0.880797 0.119203] [0.119203 0.880797]]
    print("log-likelihood:", round(log_l, 6))  # -2.970315
