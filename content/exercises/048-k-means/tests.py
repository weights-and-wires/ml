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


def labelled_blobs(seed: int = 3, n_per: int = 20):
    rng = np.random.default_rng(seed)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = np.concatenate([s + 0.8 * rng.normal(size=(n_per, 2)) for s in spots])
    blob_of = np.repeat(np.arange(3), n_per)
    return X, blob_of


@check("kmeans_objective gives 25 and 1 on the chapter's groupings, and is a sum on random data")
def test_objective(sub):
    objective = has(sub, "kmeans_objective")
    X, centers = four_points()
    got = objective(X, centers, np.array([0, 1, 1, 1]))
    if not isinstance(got, float):
        fail(f"expected a Python float, got {type(got).__name__}")
    almost(got, 25.0, tol=1e-12, label="J for centres 0 and 1")
    almost(objective(X, np.array([[0.5], [4.5]]), np.array([0, 0, 1, 1])), 1.0, tol=1e-12,
           label="J for centres 0.5 and 4.5")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    centers = rng.normal(size=(4, 3))
    assignments = rng.integers(0, 4, size=30)
    want = sum(float(np.sum((X[i] - centers[assignments[i]]) ** 2)) for i in range(30))
    almost(objective(X, centers, assignments), want, tol=1e-9, label="J")


@check("assign_nearest picks the nearest centre on the four points and agrees with a brute-force search")
def test_assign(sub):
    assign = has(sub, "assign_nearest")
    X, centers = four_points()
    got = assign(X, centers)
    shape(got, (4,), label="assignments")
    equal(np.asarray(got), np.array([0, 1, 1, 1]), label="nearest centres")
    equal(np.asarray(assign(X, np.array([[0.5], [4.5]]))), np.array([0, 0, 1, 1]),
          label="nearest of 0.5 and 4.5")
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


@check("lloyd keeps the old centre of an empty group, and leaves the init array alone")
def test_lloyd_empty_and_copy(sub):
    lloyd = has(sub, "lloyd")
    X, _ = four_points()
    init = np.array([[0.0], [1.0], [100.0]])
    before = init.copy()
    centers, assignments, history = lloyd(X, 3, init, 10)
    if not np.all(np.isfinite(centers)):
        fail(f"a centre became nan or infinite: {centers.ravel()}")
    close(np.asarray(centers)[2], np.array([100.0]), atol=1e-12, label="the empty group's centre")
    if not np.all(np.isfinite(history)):
        fail(f"the history holds a non-finite value: {history}")
    equal(init, before, label="init after lloyd")


@check("seeding_probabilities turns 0, 1, 100 into 0, 1/101, 100/101, and falls back to uniform at zero")
def test_probabilities(sub):
    probs = has(sub, "seeding_probabilities")
    got = np.asarray(probs(np.array([0.0, 1.0, 100.0])))
    shape(got, (3,), label="probabilities")
    close(got, np.array([0.0, 1 / 101, 100 / 101]), atol=1e-12, label="probabilities")
    close(got.sum(), 1.0, atol=1e-12, label="sum of the probabilities")
    zero = np.asarray(probs(np.zeros(4)))
    if not np.all(np.isfinite(zero)):
        fail(f"dividing by a zero sum gave non-finite values: {zero}")
    close(zero, np.full(4, 0.25), atol=1e-12, label="uniform fallback")


@check("kmeans_plusplus_init returns k distinct rows of X, and survives data where every point is the same")
def test_init_shape(sub):
    init = has(sub, "kmeans_plusplus_init")
    X, _ = labelled_blobs()
    got = np.asarray(init(X, 3, np.random.default_rng(0)))
    shape(got, (3, 2), label="centres")
    for row in got:
        if not np.any(np.all(np.isclose(X, row), axis=1)):
            fail(f"the centre {row} is not a row of X")
    if len({tuple(np.round(r, 9)) for r in got}) < 3:
        fail("two seeds are the same point, but a point at distance 0 has probability 0")
    same = np.asarray(init(np.ones((5, 2)), 3, np.random.default_rng(0)))
    shape(same, (3, 2), label="centres on identical points")
    if not np.all(np.isfinite(same)):
        fail(f"the centres are not finite: {same}")


@check("kmeans_plusplus_init draws the far point about 99 percent of the time on 0, 1, 10")
def test_init_distribution(sub):
    init = has(sub, "kmeans_plusplus_init")
    X = np.array([[0.0], [1.0], [10.0]])
    hits = 0
    trials = 1500
    for seed in range(trials):
        pair = np.asarray(init(X, 2, np.random.default_rng(seed))).ravel()
        hits += int(np.any(np.isclose(pair, 10.0)))
    # P(10 is in the pair) = (1/3)(100/101) + (1/3)(81/82) + 1/3, about 0.9926.
    # Plain distances instead of squares would give about 0.90.
    rate = hits / trials
    almost(rate, 0.9926, tol=0.012, label="share of seeds that include the far point")


@check("kmeans_plusplus_init spreads seeds over the three blobs far more often than uniform")
def test_init_spread(sub):
    init = has(sub, "kmeans_plusplus_init")
    X, blob_of = labelled_blobs()
    pp = 0
    uniform = 0
    trials = 60
    for seed in range(trials):
        pair = np.asarray(init(X, 3, np.random.default_rng(100 + seed)))
        idx = [int(np.argmin(np.sum((X - c) ** 2, axis=1))) for c in pair]
        pp += int(len(set(blob_of[idx])) == 3)
        idx_u = np.random.default_rng(500 + seed).choice(len(X), size=3, replace=False)
        uniform += int(len(set(blob_of[idx_u])) == 3)
    if pp < 0.65 * trials:
        fail(f"only {pp} of {trials} seedings put one seed in each blob; expected at least {int(0.65 * trials)}")
    if pp <= uniform:
        fail(f"seeding by squared distance ({pp}) should beat uniform picks ({uniform})")


@check("best_of_restarts returns the lowest J of its restarts, and finds the good grouping of three blobs")
def test_best_of(sub):
    best = has(sub, "best_of_restarts")
    best = has(sub, "best_of_restarts")
    objective = has(sub, "kmeans_objective")
    X, _ = labelled_blobs(seed=4)
    centers, assignments, j = best(X, 3, 8, np.random.default_rng(7))
    shape(np.asarray(centers), (3, 2), label="centres")
    shape(np.asarray(assignments), (len(X),), label="assignments")
    almost(float(j), objective(X, centers, assignments), tol=1e-8, label="returned J matches its groups")
    # The first restart of this call is exactly the run with this seed, so the
    # best of eight cannot be worse than it.
    _, _, j_one = best(X, 3, 1, np.random.default_rng(7))
    if j > j_one + 1e-9:
        fail(f"the best of 8 restarts has J = {j:.3f}, worse than the first restart alone ({j_one:.3f})")
    # Every restart must draw from the generator you pass in. A restart that
    # builds a fresh generator inside the loop leaves this one untouched.
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    best(X, 3, 3, rng)
    if rng.bit_generator.state == before:
        fail("the generator you passed in was never used, so every restart repeats the same start. "
             "Draw each seeding from the rng argument, and do not create a new generator inside the loop.")
    X3, _ = labelled_blobs(seed=3)
    _, _, j3 = best(X3, 3, 10, np.random.default_rng(1))
    # The planted grouping has J close to 88.5; a stuck run sits above 400.
    if j3 > 120.0:
        fail(f"J = {j3:.1f} after 10 restarts; the planted grouping gives about 88.5")
