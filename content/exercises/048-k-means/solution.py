"""Reference solution: k-means with Lloyd's steps and k-means++ seeding."""

import numpy as np


def kmeans_objective(X: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    gaps = X - centers[assignments]
    return float(np.sum(gaps * gaps))


def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2. The ||x||^2 term is the same for
    # every centre, so it cannot change which centre wins and is dropped.
    scores = np.sum(centers**2, axis=1)[None, :] - 2.0 * (X @ centers.T)
    return np.argmin(scores, axis=1)


def lloyd(X: np.ndarray, k: int, init: np.ndarray, iters: int):
    centers = init.copy()
    assignments = assign_nearest(X, centers)
    history = [kmeans_objective(X, centers, assignments)]
    for _ in range(iters):
        for j in range(k):
            members = X[assignments == j]
            if len(members) > 0:  # an empty group keeps its old centre
                centers[j] = members.mean(axis=0)
        assignments = assign_nearest(X, centers)
        history.append(kmeans_objective(X, centers, assignments))
        if history[-1] == history[-2]:
            break
    return centers, assignments, history


def seeding_probabilities(sq_dist: np.ndarray) -> np.ndarray:
    total = sq_dist.sum()
    if total <= 0.0:
        # Every point sits on a chosen centre, so no point is better than another.
        return np.full(len(sq_dist), 1.0 / len(sq_dist))
    return sq_dist / total


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


def best_of_restarts(X: np.ndarray, k: int, n_init: int, rng: np.random.Generator):
    best = None
    for _ in range(n_init):
        init = kmeans_plusplus_init(X, k, rng)
        centers, assignments, history = lloyd(X, k, init, 100)
        if best is None or history[-1] < best[2]:
            best = (centers, assignments, history[-1])
    return best


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
