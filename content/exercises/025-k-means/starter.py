"""k-means: Lloyd's steps and a good start.

Fill in the six functions below, in this order. `lloyd` returns the history of
the objective J, so you can watch it fall. The seeding functions give it a start
that is likely to be a good one, and `best_of_restarts` keeps the best of several.
"""

import numpy as np


def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    """Return J, the sum of squared gaps from each point to its own centre.

    Args:
        X: Data points, shape (n, d).
        centers: Cluster centres, shape (k, d).
        assignments: Integer centre index for each point, shape (n,), values in
            range(k).

    Returns:
        The sum over all points of the squared distance to centers[assignments[i]],
        as a Python float. It is a sum, not a mean.
    """
    raise NotImplementedError("kmeans_objective")


def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return the index of the nearest centre for every point.

    Args:
        X: Data points, shape (n, d).
        centers: Cluster centres, shape (k, d).

    Returns:
        Integer array of shape (n,) with values in range(k).

    Use ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 and drop the ||x||^2 term, which
    is the same for every centre. The scores you compare can then be negative.
    """
    raise NotImplementedError("assign_nearest")


def lloyd(X: np.ndarray, k: int, init: np.ndarray, iters: int):
    """Run Lloyd's algorithm from a fixed start.

    Steps:
        1. Copy init, assign every point to its nearest centre, and record J.
        2. Repeat up to iters times: move each centre to the mean of its group,
           reassign every point, and append J to the history.
        3. Stop early when J is exactly the same as the previous entry.

    Args:
        X: Data points, shape (n, d).
        k: Number of clusters.
        init: Starting centres, shape (k, d). Do not modify it.
        iters: Maximum number of update-then-reassign passes.

    Returns:
        A tuple (centers, assignments, history):
            centers: Final centres, shape (k, d).
            assignments: Final assignment of each point, shape (n,).
            history: List of J after the first assignment and after every pass.
                It never rises.

    A group with no points keeps its old centre, because the mean of zero points
    is nan.
    """
    raise NotImplementedError("lloyd")


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
    # The chapter's four points on a line, starting from centres 0 and 1.
    X = np.array([[0.0], [1.0], [4.0], [5.0]])
    init = np.array([[0.0], [1.0]])
    print("J for the first grouping:", kmeans_objective(X, init, np.array([0, 1, 1, 1])))  # 25.0
    print("nearest centres:", assign_nearest(X, init))  # [0 1 1 1]
    centers, assignments, history = lloyd(X, 2, init, iters=10)
    print("centres:", centers.ravel())  # [0.5 4.5]
    print("history:", [round(h, 4) for h in history])  # [25.0, 4.2222, 1.0, 1.0]

    # The running example: three clusters of delivery addresses.
    rng = np.random.default_rng(0)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    pts = np.concatenate([s + 0.8 * rng.normal(size=(8, 2)) for s in spots])
    start = pts[[0, 1, 8]]  # a deliberately poor start: two centres in one cluster
    _, _, hist = lloyd(pts, 3, start, iters=20)
    print("J by pass on the addresses:", [round(h, 2) for h in hist])

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
