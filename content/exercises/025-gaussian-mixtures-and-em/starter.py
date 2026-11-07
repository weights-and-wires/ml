"""Gaussian mixtures and EM.

A Gaussian mixture gives every point a share in every blob. The E-step
computes those shares from the log density of each blob, with a log-sum-exp
that survives scores near -1000. The M-step refits each blob as a share-weighted
mean and covariance, the lower bound explains why the log-likelihood never
falls, and the loop that alternates the two is EM. You write six functions, in
this order:

1. `log_gaussian`, 2. `logsumexp_rows`, 3. `gmm_e_step`,
4. `gmm_m_step`, 5. `gmm_elbo`, 6. `fit_gmm`.

The k-means helpers are written for you, because `fit_gmm` seeds its blobs with
`kmeans_plusplus_init`.
"""

import numpy as np


# Written in the k-means chapter.
def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    gaps = X - centers[assignments]
    return float(np.sum(gaps * gaps))


# Written in the k-means chapter.
def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    scores = np.sum(centers**2, axis=1)[None, :] - 2.0 * (X @ centers.T)
    return np.argmin(scores, axis=1)


# Written in the k-means chapter.
def lloyd(X: np.ndarray, k: int, init: np.ndarray, iters: int):
    centers = init.copy()
    assignments = assign_nearest(X, centers)
    history = [kmeans_objective(X, centers, assignments)]
    for _ in range(iters):
        for j in range(k):
            members = X[assignments == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
        assignments = assign_nearest(X, centers)
        history.append(kmeans_objective(X, centers, assignments))
        if history[-1] == history[-2]:
            break
    return centers, assignments, history


# Written in the k-means chapter.
def seeding_probabilities(sq_dist: np.ndarray) -> np.ndarray:
    total = sq_dist.sum()
    if total <= 0.0:
        return np.full(len(sq_dist), 1.0 / len(sq_dist))
    return sq_dist / total


# Written in the k-means chapter.
def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    sq_dist = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        chosen = rng.choice(n, p=seeding_probabilities(sq_dist))
        centers[j] = X[chosen]
        sq_dist = np.minimum(sq_dist, np.sum((X - centers[j]) ** 2, axis=1))
    return centers


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


def gmm_m_step(X: np.ndarray, responsibilities: np.ndarray, reg_covar: float = 1e-6):
    """Refit every blob from the responsibilities.

    Each point counts for blob j in proportion to its responsibility. Let
    N_j be the sum of column j of the responsibilities, the blob's soft count.

    Args:
        X: Points, shape (n, d).
        responsibilities: Shape (n, k), each row adding up to 1.
        reg_covar: A multiple of the identity added to every covariance.

    Returns:
        A tuple (means, covs, weights) with shapes (k, d), (k, d, d), and (k,):
            weights[j] = N_j / n
            means[j] = the responsibility-weighted average of the points
            covs[j] = the responsibility-weighted average of
                (x_i - means[j]) (x_i - means[j])^T, around the NEW mean,
                plus reg_covar * identity.
    """
    raise NotImplementedError("gmm_m_step")


def gmm_elbo(X: np.ndarray, responsibilities: np.ndarray, means: np.ndarray,
             covs: np.ndarray, weights: np.ndarray) -> float:
    """Return the lower bound on log L that the given shares produce.

    The bound is the sum over points i and blobs j of
        r_ij * (log weights[j] + log N(x_i; means[j], covs[j]) - log r_ij).

    Args:
        X: Points, shape (n, d).
        responsibilities: The shares r, shape (n, k). Some can be exactly 0.
        means, covs, weights: The blob parameters, as in gmm_e_step.

    Returns:
        The bound as a Python float. A share of 0 contributes 0, so take the log
        of r only where r is positive.
    """
    raise NotImplementedError("gmm_elbo")


def fit_gmm(X: np.ndarray, k: int, iters: int, rng: np.random.Generator, reg_covar: float = 1e-6):
    """Fit a Gaussian mixture by EM.

    Start the means at k-means++ seeds, give every blob the covariance
    np.var(X, axis=0).mean() times the identity, and give every blob the weight
    1/k. Then run the E-step, record log L, and repeat iters times:
    M-step, E-step, record log L.

    Args:
        X: Points, shape (n, d).
        k: Number of blobs.
        iters: Number of M-step then E-step rounds.
        rng: A seeded numpy random generator, used for the seeding.
        reg_covar: Passed to gmm_m_step.

    Returns:
        A tuple (means, covs, weights, responsibilities, history), where history
        holds log L after the first E-step and after each round, so it has length
        iters + 1 and never falls.
    """
    raise NotImplementedError("fit_gmm")


if __name__ == "__main__":
    # The chapter's two points, and two blobs with means 0 and 2.
    X = np.array([[0.0], [2.0]])
    means = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    weights = np.array([0.5, 0.5])

    print("log density at 0 and 2:", log_gaussian(X, means[0], covs[0]))  # [-0.9189 -2.9189]
    print("log-sum-exp of -1000, -1001:", logsumexp_rows(np.array([[-1000.0, -1001.0]])))  # about [-999.6867]
    resp, log_l = gmm_e_step(X, means, covs, weights)
    print("responsibilities:\n", np.round(resp, 6))  # [[0.880797 0.119203] [0.119203 0.880797]]
    print("log-likelihood:", round(log_l, 6))  # -2.970315

    # The M-step refits the blobs from those shares.
    new_means, new_covs, new_weights = gmm_m_step(X, resp, reg_covar=0.0)
    print("weights:", new_weights)  # [0.5 0.5]
    print("means:", new_means.ravel())  # [0.2384 1.7616]
    print("variances:", new_covs.ravel())  # [0.42 0.42]
    _, log_l_after = gmm_e_step(X, new_means, new_covs, new_weights)
    print("log L before and after:", round(log_l, 3), round(log_l_after, 3))  # -2.970, -2.439

    # The running example: three clusters, started from deliberately round blobs.
    rng = np.random.default_rng(3)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    data = np.concatenate([s + 0.8 * rng.normal(size=(20, 2)) for s in spots])
    *_, history = fit_gmm(data, 3, iters=10, rng=np.random.default_rng(5))
    print("log L by round:", [round(h, 1) for h in history])
