"""Lloyd's algorithm, k-means++, and EM for a Gaussian mixture.

Fill in the six functions below. `lloyd` and `fit_gmm` each return a history of
the objective they drive down, or the likelihood they drive up, so you can see
convergence directly.
"""

import numpy as np


def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    """Return the k-means objective for a given assignment.

    Args:
        X: Data points, shape (n, d).
        centers: Cluster centers, shape (k, d).
        assignments: Integer cluster index for each point, shape (n,), values
            in range(k).

    Returns:
        sum_i ||x_i - centers[assignments[i]]||^2, as a float.
    """
    raise NotImplementedError("kmeans_objective")


def lloyd(X: np.ndarray, k: int, init: np.ndarray, iters: int):
    """Run Lloyd's algorithm from a fixed initialisation.

    Alternate the two half-steps: assign every point to its nearest center,
    then move every center to the mean of the points assigned to it. Stop
    early if an iteration leaves the assignment unchanged.

    Args:
        X: Data points, shape (n, d).
        k: Number of clusters.
        init: Initial centers, shape (k, d).
        iters: Maximum number of assign-then-update iterations to run.

    Returns:
        A tuple (centers, assignments, history):
            centers: Final centers, shape (k, d).
            assignments: Final assignment for each point, shape (n,).
            history: List of the objective value after the initial assignment
                and after every subsequent iteration, so it has length at
                most iters + 1 and is non-increasing.

    An empty cluster (no points assigned to it) keeps its previous center
    rather than producing a NaN mean.
    """
    raise NotImplementedError("lloyd")


def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seed k centers with the k-means++ distribution.

    Choose the first center uniformly at random among the data points. Choose
    every subsequent center from the remaining points with probability
    proportional to its squared distance to the nearest center chosen so far.

    Args:
        X: Data points, shape (n, d).
        k: Number of centers to choose.
        rng: A seeded numpy random generator, used for every random choice.

    Returns:
        The chosen centers, shape (k, d), each an actual row of X.
    """
    raise NotImplementedError("kmeans_plusplus_init")


def gmm_e_step(X: np.ndarray, means: np.ndarray, covs: np.ndarray, weights: np.ndarray):
    """Compute responsibilities and the incomplete-data log-likelihood.

    Args:
        X: Data points, shape (n, d).
        means: Component means, shape (k, d).
        covs: Component covariances, shape (k, d, d), each positive definite.
        weights: Mixing weights, shape (k,), summing to one.

    Returns:
        A tuple (responsibilities, log_likelihood):
            responsibilities: shape (n, k), row j sums to one; entry (i, j) is
                p(z_i = j | x_i, theta).
            log_likelihood: sum_i log p(x_i | theta), as a float.

    Compute this in the log domain and subtract the per-row maximum before
    exponentiating, so it stays finite when the components are far apart.
    """
    raise NotImplementedError("gmm_e_step")


def gmm_m_step(X: np.ndarray, responsibilities: np.ndarray, reg_covar: float = 1e-6):
    """Compute the closed-form maximum-likelihood update given responsibilities.

    Args:
        X: Data points, shape (n, d).
        responsibilities: shape (n, k), from the E-step.
        reg_covar: Multiple of the identity added to every covariance, to keep
            it invertible when a component's effective sample size is small.

    Returns:
        A tuple (means, covs, weights) with shapes (k, d), (k, d, d), (k,).
    """
    raise NotImplementedError("gmm_m_step")


def fit_gmm(X: np.ndarray, k: int, iters: int, rng: np.random.Generator, reg_covar: float = 1e-6):
    """Fit a Gaussian mixture by EM, initialised with k-means++ and a shared variance.

    Args:
        X: Data points, shape (n, d).
        k: Number of mixture components.
        iters: Number of EM iterations to run after the initial E-step.
        rng: A seeded numpy random generator, used for the k-means++ seeding.
        reg_covar: Passed through to gmm_m_step.

    Returns:
        A tuple (means, covs, weights, responsibilities, ll_history):
            means, covs, weights: The fitted parameters, as in gmm_m_step.
            responsibilities: The final E-step responsibilities, shape (n, k).
            ll_history: List of the log-likelihood after the initial E-step and
                after every subsequent EM iteration; it has length iters + 1
                and is non-decreasing.

    Initialise the means with kmeans_plusplus_init, every covariance with the
    mean per-feature variance of X times the identity, and the weights
    uniformly.
    """
    raise NotImplementedError("fit_gmm")


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
