"""Checks for the leaves and impurity exercise."""

import numpy as np

from mlcheck import almost, check, close, fail

EIGHT_HOUSES = np.array([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0])


@check("gini is 0 for a pure box, 0.75 for four equal classes, and 22/36 for [0, 0, 1, 1, 1, 2]")
def test_gini(sub):
    almost(sub.gini(np.array([2, 2, 2, 2])), 0.0, tol=1e-12, label="pure box")
    almost(sub.gini(np.array([0, 1, 2, 3])), 0.75, tol=1e-12, label="four equal classes")
    almost(sub.gini(np.array([0, 0, 1, 1, 1, 2])), 22 / 36, tol=1e-12, label="mixed box")
    almost(sub.gini(np.array([0.0, 1.0, 1.0])), 4 / 9, tol=1e-12, label="float labels")
    almost(sub.gini(np.array([], dtype=int)), 0.0, tol=1e-12, label="empty box")


@check("gini of a 0/1 box with a fraction p of ones is 2p(1 - p)")
def test_gini_binary(sub):
    for ones in range(0, 11):
        y = np.array([1] * ones + [0] * (10 - ones))
        p = ones / 10
        almost(sub.gini(y), 2 * p * (1 - p), tol=1e-12, label=f"{ones} ones out of 10")


@check("entropy is 0 for a pure box, log 4 for four equal classes, and 1.0114 for [0, 0, 1, 1, 1, 2]")
def test_entropy(sub):
    almost(sub.entropy(np.array([1, 1, 1, 1])), 0.0, tol=1e-12, label="pure box")
    almost(sub.entropy(np.array([0, 1, 2, 3])), np.log(4.0), tol=1e-12, label="four equal classes")
    p = np.array([2, 3, 1]) / 6
    almost(sub.entropy(np.array([0, 0, 1, 1, 1, 2])), -np.sum(p * np.log(p)), tol=1e-12, label="mixed box")
    almost(sub.entropy(np.array([0, 0, 1, 1, 1, 2])), 1.0114, tol=1e-4, label="the number in the chapter")


@check("entropy skips classes that have no labels, so it never returns nan")
def test_entropy_gaps(sub):
    got = sub.entropy(np.array([0, 0, 3, 3]))
    if not np.isfinite(got):
        fail(f"entropy of [0, 0, 3, 3] is {got}; classes 1 and 2 have no labels and must add nothing")
    almost(got, np.log(2.0), tol=1e-12, label="two equal classes with a gap between them")
    almost(sub.entropy(np.array([], dtype=int)), 0.0, tol=1e-12, label="empty box")


@check("variance_impurity is the mean squared gap to the mean: 13.5 for the eight houses")
def test_variance(sub):
    almost(sub.variance_impurity(EIGHT_HOUSES), 13.5, tol=1e-12, label="eight houses")
    almost(sub.variance_impurity(np.array([1.0, 3.0, 8.0, 10.0])), 13.25, tol=1e-12, label="four houses")
    almost(sub.variance_impurity(np.array([7.0, 7.0, 7.0])), 0.0, tol=1e-12, label="a constant box")
    almost(sub.variance_impurity(np.array([])), 0.0, tol=1e-12, label="empty box")
    almost(sub.variance_impurity(EIGHT_HOUSES + 100.0), 13.5, tol=1e-9, label="shifted by 100")


@check("for 0/1 labels, variance is p(1 - p) and gini is exactly twice the variance")
def test_three_costumes(sub):
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=37) < 0.3).astype(float)
    p = y.mean()
    almost(sub.variance_impurity(y), p * (1 - p), tol=1e-12, label="variance of a 0/1 box")
    almost(sub.gini(y), 2 * sub.variance_impurity(y), tol=1e-12, label="gini versus twice the variance")


@check("the mean is the best single answer, and variance_impurity is the miss it leaves")
def test_leaf(sub):
    rng = np.random.default_rng(1)
    y = rng.normal(loc=5.0, scale=2.0, size=40)
    c = sub.leaf_value(y)
    miss = float(np.mean((y - c) ** 2))
    almost(sub.variance_impurity(y), miss, tol=1e-10, label="variance_impurity versus the mean squared miss at the mean")
    for shift in (-2.0, -0.5, -0.01, 0.01, 0.5, 2.0):
        other = float(np.mean((y - (c + shift)) ** 2))
        if other <= miss:
            fail(f"guessing {c + shift:.4f} misses by {other:.6f}, no worse than the mean's {miss:.6f}")
    box = np.array([1.0, 3.0])
    almost(float(np.sum((box - 2.0) ** 2)), 2.0, tol=1e-12, label="squared error of [1, 3] at 2")
