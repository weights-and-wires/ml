"""k-means++ seeding and restarts.

Fill in the three functions below. The three k-means functions are your own code
from the previous chapter, already filled in, so you can run Lloyd from any seed.
"""

import numpy as np


# Written in the chapter "k-means: group points by moving centres".
def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    gaps = X - centers[assignments]
    return float(np.sum(gaps * gaps))


# Written in the chapter "k-means: group points by moving centres".
def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    scores = np.sum(centers**2, axis=1)[None, :] - 2.0 * (X @ centers.T)
    return np.argmin(scores, axis=1)


# Written in the chapter "k-means: group points by moving centres".
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


def seeding_probabilities(sq_dist: np.ndarray) -> np.ndarray:
    """Turn squared distances into the probability of picking each point.

    Args:
        sq_dist: Squared distance from each point to its nearest centre chosen
            so far, shape (n,), all at least 0.

    Returns:
        Probabilities of shape (n,) that add up to 1, each proportional to its
        squared distance. When every squared distance is 0, return the uniform
        probabilities 1/n, because dividing by a zero sum gives nan.
    """
    raise NotImplementedError("seeding_probabilities")


def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seed k centres with the k-means++ rule.

    Pick the first centre uniformly from the points. Pick each later centre from
    the points with probability proportional to its squared distance to the
    nearest centre chosen so far.

    Args:
        X: Data points, shape (n, d).
        k: Number of centres to choose.
        rng: A seeded numpy random generator. Use it for every random choice.

    Returns:
        The chosen centres, shape (k, d). Each is a row of X.

    Keep one array of squared distances to the nearest chosen centre, and shrink
    it with np.minimum after each pick, instead of recomputing from scratch.
    """
    raise NotImplementedError("kmeans_plusplus_init")


def best_of_restarts(X: np.ndarray, k: int, n_init: int, rng: np.random.Generator):
    """Run seeding plus Lloyd n_init times and keep the lowest-J run.

    Args:
        X: Data points, shape (n, d).
        k: Number of clusters.
        n_init: Number of restarts, at least 1.
        rng: A seeded numpy random generator, passed to kmeans_plusplus_init.

    Returns:
        A tuple (centers, assignments, J) for the restart whose final J is
        smallest, where J is the last entry of that run's history. Run lloyd
        with up to 100 passes.
    """
    raise NotImplementedError("best_of_restarts")


if __name__ == "__main__":
    # The chapter's three points on a line, with the first centre at 0.
    sq_dist = np.array([0.0, 1.0, 100.0])
    print("seeding probabilities:", np.round(seeding_probabilities(sq_dist), 4))  # [0, 0.0099, 0.9901]
    print("all on a centre:", seeding_probabilities(np.zeros(3)))  # [1/3 1/3 1/3]

    # Three clusters of delivery addresses, 20 per cluster.
    rng = np.random.default_rng(3)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = np.concatenate([s + 0.8 * rng.normal(size=(20, 2)) for s in spots])

    seeds = kmeans_plusplus_init(X, 3, np.random.default_rng(0))
    print("seeded centres:\n", np.round(seeds, 2))
    _, _, history = lloyd(X, 3, seeds, 50)
    print("J after Lloyd from that seed:", round(history[-1], 1))

    _, _, best_j = best_of_restarts(X, 3, 10, np.random.default_rng(1))
    print("best J over 10 restarts:", round(best_j, 1))
