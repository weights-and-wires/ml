"""Reference solution for the clustering and EM exercise."""

import numpy as np


def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    diffs = X - centers[assignments]
    return float(np.sum(diffs * diffs))


def _assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2; the ||x||^2 term does not affect
    # the argmin over centers, so it is dropped.
    dots = X @ centers.T
    sq_norms = np.sum(centers * centers, axis=1)
    return np.argmin(sq_norms[None, :] - 2.0 * dots, axis=1)


def lloyd(X: np.ndarray, k: int, init: np.ndarray, iters: int):
    centers = init.copy()
    assignments = _assign(X, centers)
    history = [kmeans_objective(X, centers, assignments)]
    for _ in range(iters):
        new_centers = centers.copy()
        for j in range(k):
            members = X[assignments == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
            # An empty cluster keeps its old center rather than producing a NaN.
        new_assignments = _assign(X, new_centers)
        centers = new_centers
        assignments = new_assignments
        history.append(kmeans_objective(X, centers, assignments))
        if history[-1] == history[-2]:
            break
    return centers, assignments, history


def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    first = rng.integers(n)
    centers[0] = X[first]
    sq_dist = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = sq_dist.sum()
        if total <= 0.0:
            # Every remaining point coincides with a chosen center.
            probs = np.full(n, 1.0 / n)
        else:
            probs = sq_dist / total
        chosen = rng.choice(n, p=probs)
        centers[j] = X[chosen]
        new_sq_dist = np.sum((X - centers[j]) ** 2, axis=1)
        sq_dist = np.minimum(sq_dist, new_sq_dist)
    return centers


def _log_gaussian(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    precision = np.linalg.inv(cov)
    maha = np.einsum("ij,jk,ik->i", diff, precision, diff)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)


def gmm_e_step(X: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray):
    n, k = X.shape[0], means.shape[0]
    log_weighted = np.empty((n, k))
    for j in range(k):
        log_weighted[:, j] = np.log(weights[j]) + _log_gaussian(X, means[j], covs[j])
    max_log = log_weighted.max(axis=1, keepdims=True)
    shifted = np.exp(log_weighted - max_log)
    row_sums = shifted.sum(axis=1, keepdims=True)
    responsibilities = shifted / row_sums
    log_likelihood = float(np.sum(max_log[:, 0] + np.log(row_sums[:, 0])))
    return responsibilities, log_likelihood


def gmm_m_step(X: np.ndarray, responsibilities: np.ndarray, reg_covar: float = 1e-6):
    n, d = X.shape
    k = responsibilities.shape[1]
    n_k = responsibilities.sum(axis=0)
    weights = n_k / n
    means = (responsibilities.T @ X) / n_k[:, None]
    covs = np.empty((k, d, d))
    for j in range(k):
        diff = X - means[j]
        weighted = responsibilities[:, j][:, None] * diff
        covs[j] = (weighted.T @ diff) / n_k[j] + reg_covar * np.eye(d)
    return means, covs, weights


def fit_gmm(X: np.ndarray, k: int, iters: int, rng: np.random.Generator, reg_covar: float = 1e-6):
    n, d = X.shape
    means = kmeans_plusplus_init(X, k, rng)
    var = np.var(X, axis=0).mean()
    covs = np.stack([var * np.eye(d) for _ in range(k)])
    weights = np.full(k, 1.0 / k)

    responsibilities, ll = gmm_e_step(X, means, covs, weights)
    ll_history = [ll]
    for _ in range(iters):
        means, covs, weights = gmm_m_step(X, responsibilities, reg_covar=reg_covar)
        responsibilities, ll = gmm_e_step(X, means, covs, weights)
        ll_history.append(ll)
    return means, covs, weights, responsibilities, ll_history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = np.concatenate(
        [center + 0.7 * rng.normal(size=(80, 2)) for center in true_centers]
    )

    init = kmeans_plusplus_init(X, 3, rng)
    centers, assignments, history = lloyd(X, 3, init, iters=20)
    print(f"Lloyd's objective by iteration: {[round(h, 2) for h in history]}")

    means, covs, weights, resp, ll_history = fit_gmm(X, 3, iters=15, rng=np.random.default_rng(1))
    print(f"EM log-likelihood by iteration: {[round(v, 2) for v in ll_history]}")
    print(f"final mixture weights: {np.round(weights, 3)}")
