"""Reference solution: the E-step of a Gaussian mixture."""

import numpy as np


def log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    gap = X - mean
    _, logdet = np.linalg.slogdet(cov)
    # Row i of (gap @ inv(cov)) times row i of gap, added up: the squared
    # distance from the mean, measured with the covariance.
    squared = np.sum((gap @ np.linalg.inv(cov)) * gap, axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + squared)


def logsumexp_rows(s: np.ndarray) -> np.ndarray:
    biggest = s.max(axis=1, keepdims=True)
    return biggest[:, 0] + np.log(np.exp(s - biggest).sum(axis=1))


def gmm_e_step(X: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray):
    n, k = X.shape[0], means.shape[0]
    scores = np.empty((n, k))
    for j in range(k):
        scores[:, j] = np.log(weights[j]) + log_gaussian(X, means[j], covs[j])
    log_px = logsumexp_rows(scores)
    responsibilities = np.exp(scores - log_px[:, None])
    return responsibilities, float(log_px.sum())


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
