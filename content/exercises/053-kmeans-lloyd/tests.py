"""Checks for the k-means exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, has, shape


def four_points():
    X = np.array([[0.0], [1.0], [4.0], [5.0]])
    init = np.array([[0.0], [1.0]])
    return X, init


def blobs(seed: int = 0, n_per: int = 20):
    rng = np.random.default_rng(seed)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    return np.concatenate([s + 0.8 * rng.normal(size=(n_per, 2)) for s in spots])


@check("kmeans_objective gives 25 for the chapter's first grouping")
def test_objective_hand(sub):
    objective = has(sub, "kmeans_objective")
    X, centers = four_points()
    got = objective(X, centers, np.array([0, 1, 1, 1]))
    if not isinstance(got, float):
        fail(f"expected a Python float, got {type(got).__name__}")
    almost(got, 25.0, tol=1e-12, label="J for centres 0 and 1")
    almost(objective(X, np.array([[0.5], [4.5]]), np.array([0, 0, 1, 1])), 1.0, tol=1e-12,
           label="J for centres 0.5 and 4.5")


@check("kmeans_objective is a sum, not a mean, on random data")
def test_objective_random(sub):
    objective = has(sub, "kmeans_objective")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    centers = rng.normal(size=(4, 3))
    assignments = rng.integers(0, 4, size=30)
    want = sum(float(np.sum((X[i] - centers[assignments[i]]) ** 2)) for i in range(30))
    almost(objective(X, centers, assignments), want, tol=1e-9, label="J")


@check("assign_nearest picks the nearest centre on the four points")
def test_assign_hand(sub):
    assign = has(sub, "assign_nearest")
    X, centers = four_points()
    got = assign(X, centers)
    shape(got, (4,), label="assignments")
    equal(np.asarray(got), np.array([0, 1, 1, 1]), label="nearest centres")
    equal(np.asarray(assign(X, np.array([[0.5], [4.5]]))), np.array([0, 0, 1, 1]),
          label="nearest of 0.5 and 4.5")


@check("assign_nearest agrees with a brute-force search in 3 dimensions")
def test_assign_random(sub):
    assign = has(sub, "assign_nearest")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3)) * 3
    centers = rng.normal(size=(5, 3)) * 3
    want = np.array([np.argmin([np.sum((x - c) ** 2) for c in centers]) for x in X])
    got = np.asarray(assign(X, centers))
    shape(got, (200,), label="assignments")
    if not np.array_equal(got, want):
        fail(f"{int(np.sum(got != want))} of 200 points went to a centre that is not the nearest")


@check("lloyd reproduces the chapter's history 25, 38/9, 1, 1")
def test_lloyd_hand(sub):
    lloyd = has(sub, "lloyd")
    X, init = four_points()
    centers, assignments, history = lloyd(X, 2, init, 10)
    close(np.array(history), np.array([25.0, 38 / 9, 1.0, 1.0]), atol=1e-12, label="history")
    close(np.asarray(centers).ravel(), np.array([0.5, 4.5]), atol=1e-12, label="final centres")
    equal(np.asarray(assignments), np.array([0, 0, 1, 1]), label="final assignments")


@check("lloyd's J never rises, and the last entry matches the returned groups")
def test_lloyd_monotone(sub):
    lloyd = has(sub, "lloyd")
    objective = has(sub, "kmeans_objective")
    X = blobs(seed=3)
    init = X[np.random.default_rng(5).choice(len(X), size=3, replace=False)]
    centers, assignments, history = lloyd(X, 3, init, 25)
    shape(centers, (3, 2), label="centres")
    shape(assignments, (len(X),), label="assignments")
    if np.any(np.diff(history) > 1e-9):
        fail(f"J rose on some pass, which a correct assign-then-update cannot do: {[round(h, 3) for h in history]}")
    almost(objective(X, centers, assignments), history[-1], tol=1e-8, label="J of the returned groups")


@check("lloyd keeps the old centre of an empty group instead of producing nan")
def test_lloyd_empty(sub):
    lloyd = has(sub, "lloyd")
    X, _ = four_points()
    init = np.array([[0.0], [1.0], [100.0]])
    centers, assignments, history = lloyd(X, 3, init, 10)
    if not np.all(np.isfinite(centers)):
        fail(f"a centre became nan or infinite: {centers.ravel()}")
    close(np.asarray(centers)[2], np.array([100.0]), atol=1e-12, label="the empty group's centre")
    if not np.all(np.isfinite(history)):
        fail(f"the history holds a non-finite value: {history}")


@check("lloyd leaves the init array alone")
def test_lloyd_copy(sub):
    lloyd = has(sub, "lloyd")
    X, init = four_points()
    before = init.copy()
    lloyd(X, 2, init, 10)
    equal(init, before, label="init after lloyd")
