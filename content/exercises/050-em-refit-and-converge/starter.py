"""The M-step, the lower bound, and the full EM loop.

Fill in the three functions below. Everything above them is your own code from
the earlier chapters, already filled in.
"""

import numpy as np


# Written in the chapter "k-means: Lloyd's steps and a good start".
def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    gaps = X - centers[assignments]
    return float(np.sum(gaps * gaps))


# Written in the chapter "k-means: Lloyd's steps and a good start".
def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    scores = np.sum(centers**2, axis=1)[None, :] - 2.0 * (X @ centers.T)
    return np.argmin(scores, axis=1)


# Written in the chapter "k-means: Lloyd's steps and a good start".
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


# Written in the chapter "k-means: Lloyd's steps and a good start".
def seeding_probabilities(sq_dist: np.ndarray) -> np.ndarray:
    total = sq_dist.sum()
    if total <= 0.0:
        return np.full(len(sq_dist), 1.0 / len(sq_dist))
    return sq_dist / total


# Written in the chapter "k-means: Lloyd's steps and a good start".
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


# Written in the chapter "Gaussian mixtures and soft assignment".
def log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    gap = X - mean
    _, logdet = np.linalg.slogdet(cov)
    squared = np.sum((gap @ np.linalg.inv(cov)) * gap, axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + squared)


# Written in the chapter "Gaussian mixtures and soft assignment".
def logsumexp_rows(s: np.ndarray) -> np.ndarray:
    biggest = s.max(axis=1, keepdims=True)
    return biggest[:, 0] + np.log(np.exp(s - biggest).sum(axis=1))


# Written in the chapter "Gaussian mixtures and soft assignment".
def gmm_e_step(X: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray):
    n, k = X.shape[0], means.shape[0]
    scores = np.empty((n, k))
    for j in range(k):
        scores[:, j] = np.log(weights[j]) + log_gaussian(X, means[j], covs[j])
    log_px = logsumexp_rows(scores)
    responsibilities = np.exp(scores - log_px[:, None])
    return responsibilities, float(log_px.sum())


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
    # The chapter's two points. The E-step gives these shares; the M-step refits.
    X = np.array([[0.0], [2.0]])
    resp = np.array([[0.880797, 0.119203], [0.119203, 0.880797]])
    means, covs, weights = gmm_m_step(X, resp, reg_covar=0.0)
    print("weights:", weights)  # [0.5 0.5]
    print("means:", means.ravel())  # [0.2384 1.7616]
    print("variances:", covs.ravel())  # [0.42 0.42]
    log_l_before = gmm_e_step(X, np.array([[0.0], [2.0]]), np.array([[[1.0]], [[1.0]]]), np.array([0.5, 0.5]))[1]
    _, log_l_after = gmm_e_step(X, means, covs, weights)
    print("log L before and after:", round(log_l_before, 3), round(log_l_after, 3))  # -2.970, -2.439

    # The running example: three clusters, started from deliberately round blobs.
    rng = np.random.default_rng(3)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    data = np.concatenate([s + 0.8 * rng.normal(size=(20, 2)) for s in spots])
    *_, history = fit_gmm(data, 3, iters=10, rng=np.random.default_rng(5))
    print("log L by round:", [round(h, 1) for h in history])
