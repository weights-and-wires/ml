"""Checks for the gain and best split exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail

HOUSES_Y = np.array([1.0, 3.0, 8.0, 10.0])


def impurity(y, criterion):
    """An independent reference, written from the definitions."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    if criterion == "mse":
        return float(np.mean((y - y.mean()) ** 2))
    p = np.bincount(y.astype(int)) / y.size
    p = p[p > 0]
    if criterion == "gini":
        return float(1.0 - np.sum(p**2))
    return float(-np.sum(p * np.log(p)))


def brute_force(X, y, criterion, min_samples_leaf=1):
    """Try every midpoint, recomputing both impurities from scratch."""
    n, d = X.shape
    parent = impurity(y, criterion)
    best = None
    for j in range(d):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            t = 0.5 * (a + b)
            left = X[:, j] <= t
            n_left = int(left.sum())
            if n_left < min_samples_leaf or n - n_left < min_samples_leaf:
                continue
            gain = parent - (
                n_left * impurity(y[left], criterion)
                + (n - n_left) * impurity(y[~left], criterion)
            ) / n
            if best is None or gain > best[0]:
                best = (gain, j, t)
    return best


@check("split_gain scores the three cuts of the four-house example: 6.75, 12.25, 6.75")
def test_gain_hand(sub):
    almost(sub.split_gain(HOUSES_Y, np.array([True, False, False, False]), "mse"), 6.75, tol=1e-12, label="cut at 1.5")
    almost(sub.split_gain(HOUSES_Y, np.array([True, True, False, False]), "mse"), 12.25, tol=1e-12, label="cut at 2.5")
    almost(sub.split_gain(HOUSES_Y, np.array([True, True, True, False]), "mse"), 6.75, tol=1e-12, label="cut at 3.5")


@check("split_gain weights each child by its size, so peeling off one row is a weak split")
def test_gain_weights(sub):
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    peel = np.zeros(10, dtype=bool)
    peel[0] = True
    almost(sub.split_gain(y, peel, "gini"), 1 / 18, tol=1e-12, label="peel off one row")
    real = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    almost(sub.split_gain(y, real, "gini"), 0.5, tol=1e-12, label="a perfect split")
    y2 = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    almost(sub.split_gain(y2, np.arange(10) < 5, "gini"), 0.18, tol=1e-12, label="a good but imperfect split")
    almost(sub.split_gain(y, np.zeros(10, dtype=bool), "gini"), 0.0, tol=1e-12, label="an empty left child")
    almost(sub.split_gain(y, np.ones(10, dtype=bool), "gini"), 0.0, tol=1e-12, label="an empty right child")


@check("split_gain is 0 when both children keep the parent's mix, and matches the definition for every criterion")
def test_gain_criteria(sub):
    almost(sub.split_gain(np.array([0, 1, 0, 1]), np.array([True, True, False, False]), "gini"), 0.0, tol=1e-12, label="gini, same mix on both sides")
    rng = np.random.default_rng(0)
    y_class = (rng.uniform(size=30) < 0.4).astype(int)
    y_num = rng.normal(size=30)
    mask = rng.uniform(size=30) < 0.5
    for criterion, y in (("gini", y_class), ("entropy", y_class), ("mse", y_num)):
        n_left = mask.sum()
        want = impurity(y, criterion) - (
            n_left * impurity(y[mask], criterion) + (30 - n_left) * impurity(y[~mask], criterion)
        ) / 30
        almost(sub.split_gain(y, mask, criterion), want, tol=1e-12, label=f"gain under {criterion}")


@check("candidate_thresholds returns midpoints between distinct sorted values")
def test_thresholds(sub):
    got = sub.candidate_thresholds(np.array([3.0, 1.0, 2.0, 2.0, 5.0]))
    close(got, np.array([1.5, 2.5, 4.0]), atol=1e-12, label="thresholds of [3, 1, 2, 2, 5]")
    got2 = sub.candidate_thresholds(np.array([3.0, 1.0, 2.0, 2.0, 5.0]), min_samples_leaf=2)
    close(got2, np.array([2.5]), atol=1e-12, label="with at least 2 rows on each side")


@check("candidate_thresholds is empty when no cut can separate the rows, and never lands on a data value")
def test_thresholds_edge(sub):
    for x in (np.array([4.0, 4.0, 4.0]), np.array([7.0]), np.array([])):
        got = sub.candidate_thresholds(x)
        equal(np.asarray(got).shape, (0,), label=f"thresholds of {x.tolist()}")
    rng = np.random.default_rng(2)
    x = rng.normal(size=50)
    got = sub.candidate_thresholds(x)
    equal(got.shape, (49,), label="50 distinct values give 49 thresholds")
    if np.any(np.isin(got, x)):
        fail("a threshold equals a data value; it should sit halfway between two neighbours")
    if np.any(np.diff(got) <= 0):
        fail("thresholds should be in increasing order")


@check("best_split finds the cut at 2.5 with gain 12.25 on the four houses, and the right feature on separable data")
def test_best_split_hand(sub):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = sub.best_split(X, HOUSES_Y, "mse")
    if result is None:
        fail("best_split returned None on the four-house example")
    feature, threshold, gain = result
    equal(feature, 0, label="feature")
    almost(threshold, 2.5, tol=1e-12, label="threshold")
    almost(gain, 12.25, tol=1e-12, label="gain")

    rng = np.random.default_rng(3)
    X2 = np.column_stack([rng.normal(size=40), np.arange(40.0)])
    y2 = (np.arange(40) >= 25).astype(int)
    feature, threshold, gain = sub.best_split(X2, y2, "gini")
    equal(feature, 1, label="the informative feature")
    almost(threshold, 24.5, tol=1e-12, label="threshold between rows 24 and 25")
    almost(gain, impurity(y2, "gini"), tol=1e-12, label="a perfect split removes all the impurity")


@check("best_split matches a slow search that recomputes both impurities at every cut")
def test_best_split_brute(sub):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3))
    y_num = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=80)
    y_cls = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    for criterion, y, msl in (("mse", y_num, 1), ("gini", y_cls, 1), ("entropy", y_cls, 5), ("mse", y_num, 10)):
        got = sub.best_split(X, y, criterion, msl)
        want = brute_force(X, y, criterion, msl)
        if got is None:
            fail(f"best_split returned None under {criterion} with min_samples_leaf={msl}, but a split exists")
        almost(got[2], want[0], tol=1e-9, label=f"best gain under {criterion}, min_samples_leaf={msl}")
        left = X[:, got[0]] <= got[1]
        n_left = int(left.sum())
        if n_left < msl or 80 - n_left < msl:
            fail(f"the split leaves {n_left} and {80 - n_left} rows, but min_samples_leaf is {msl}")


@check("best_split returns None when no valid cut exists, and survives labels with a huge offset")
def test_best_split_none_and_offset(sub):
    X = np.ones((6, 2))
    if sub.best_split(X, np.arange(6.0), "mse") is not None:
        fail("every feature is constant, so best_split should return None")
    if sub.best_split(np.arange(6.0).reshape(-1, 1), np.arange(6.0), "mse", 4) is not None:
        fail("6 rows cannot leave 4 on each side, so best_split should return None")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 2))
    y = X[:, 0] + 0.2 * rng.normal(size=60)
    plain = sub.best_split(X, y, "mse")
    shifted = sub.best_split(X, y + 1e8, "mse")
    equal(shifted[:2], plain[:2], label="feature and threshold after adding 1e8 to every label")
    almost(shifted[2], plain[2], tol=1e-6, label="gain after adding 1e8 to every label")
