"""Reference solution: Gaussian mixtures and EM.

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
    d = X.shape[1]
    gap = X - mean
    _, logdet = np.linalg.slogdet(cov)
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


def gmm_m_step(X: np.ndarray, responsibilities: np.ndarray, reg_covar: float = 1e-6):
    n, d = X.shape
    k = responsibilities.shape[1]
    counts = responsibilities.sum(axis=0)  # N_j, the soft count of each blob
    weights = counts / n
    means = (responsibilities.T @ X) / counts[:, None]
    covs = np.empty((k, d, d))
    for j in range(k):
        gap = X - means[j]  # around the NEW mean
        covs[j] = (responsibilities[:, j, None] * gap).T @ gap / counts[j] + reg_covar * np.eye(d)
    return means, covs, weights


def gmm_elbo(X, responsibilities, means, covs, weights) -> float:
    k = means.shape[0]
    scores = np.stack(
        [np.log(weights[j]) + log_gaussian(X, means[j], covs[j]) for j in range(k)], axis=1
    )
    # r log r is 0 when r is 0, so take the log only where r is positive.
    log_r = np.log(responsibilities, out=np.zeros_like(responsibilities), where=responsibilities > 0)
    return float(np.sum(responsibilities * (scores - log_r)))


def fit_gmm(X: np.ndarray, k: int, iters: int, rng: np.random.Generator, reg_covar: float = 1e-6):
    n, d = X.shape
    means = kmeans_plusplus_init(X, k, rng)
    covs = np.stack([np.var(X, axis=0).mean() * np.eye(d) for _ in range(k)])
    weights = np.full(k, 1.0 / k)

    responsibilities, log_l = gmm_e_step(X, means, covs, weights)
    history = [log_l]
    for _ in range(iters):
        means, covs, weights = gmm_m_step(X, responsibilities, reg_covar)
        responsibilities, log_l = gmm_e_step(X, means, covs, weights)
        history.append(log_l)
    return means, covs, weights, responsibilities, history


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
