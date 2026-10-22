"""k-means by Lloyd's two half-steps.

Fill in the three functions below. `lloyd` returns the history of the objective
J, so you can watch it fall.
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
