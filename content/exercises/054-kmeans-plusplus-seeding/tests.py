"""Checks for the k-means++ seeding exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def blobs(seed: int = 3, n_per: int = 20):
    rng = np.random.default_rng(seed)
    spots = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = np.concatenate([s + 0.8 * rng.normal(size=(n_per, 2)) for s in spots])
    blob_of = np.repeat(np.arange(3), n_per)
    return X, blob_of


@check("seeding_probabilities turns squared distances 0, 1, 100 into 0, 1/101, 100/101")
def test_probabilities(sub):
    probs = has(sub, "seeding_probabilities")
    got = np.asarray(probs(np.array([0.0, 1.0, 100.0])))
    shape(got, (3,), label="probabilities")
    close(got, np.array([0.0, 1 / 101, 100 / 101]), atol=1e-12, label="probabilities")
    close(got.sum(), 1.0, atol=1e-12, label="sum of the probabilities")


@check("seeding_probabilities falls back to uniform when every distance is 0")
def test_probabilities_zero(sub):
    probs = has(sub, "seeding_probabilities")
    got = np.asarray(probs(np.zeros(4)))
    if not np.all(np.isfinite(got)):
        fail(f"dividing by a zero sum gave non-finite values: {got}")
    close(got, np.full(4, 0.25), atol=1e-12, label="uniform fallback")


@check("kmeans_plusplus_init returns k distinct rows of X")
def test_init_shape(sub):
    init = has(sub, "kmeans_plusplus_init")
    X, _ = blobs()
    got = np.asarray(init(X, 3, np.random.default_rng(0)))
    shape(got, (3, 2), label="centres")
    for row in got:
        if not np.any(np.all(np.isclose(X, row), axis=1)):
            fail(f"the centre {row} is not a row of X")
    if len({tuple(np.round(r, 9)) for r in got}) < 3:
        fail("two seeds are the same point, but a point at distance 0 has probability 0")


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
    X, blob_of = blobs()
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


@check("kmeans_plusplus_init does not break when every point is the same")
def test_init_identical(sub):
    init = has(sub, "kmeans_plusplus_init")
    X = np.ones((5, 2))
    got = np.asarray(init(X, 3, np.random.default_rng(0)))
    shape(got, (3, 2), label="centres")
    if not np.all(np.isfinite(got)):
        fail(f"the centres are not finite: {got}")


@check("best_of_restarts returns the lowest J of its restarts")
def test_best_of(sub):
    best = has(sub, "best_of_restarts")
    objective = has(sub, "kmeans_objective")
    X, _ = blobs(seed=4)
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


@check("best_of_restarts finds the good grouping of three blobs")
def test_best_of_quality(sub):
    best = has(sub, "best_of_restarts")
    X, blob_of = blobs(seed=3)
    _, assignments, j = best(X, 3, 10, np.random.default_rng(1))
    # The planted grouping has J close to 88.5; a stuck run sits above 400.
    if j > 120.0:
        fail(f"J = {j:.1f} after 10 restarts; the planted grouping gives about 88.5")
