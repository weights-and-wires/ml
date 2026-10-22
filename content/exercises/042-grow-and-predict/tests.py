"""Checks for the grow a tree exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail

HOUSES = np.array(
    [[1.0, 5.0], [2.0, 3.0], [2.5, 1.0], [3.0, 6.0], [4.0, 4.0], [5.5, 2.0], [6.0, 5.0], [7.0, 1.0]]
)
EXPENSIVE = np.array([0, 0, 1, 0, 0, 1, 1, 1])


def leaf(sub, value, n=1):
    return sub.Node(n_samples=n, impurity=0.0, value=value, is_leaf=True)


def split(sub, feature, threshold, left, right):
    n = left.n_samples + right.n_samples
    return sub.Node(
        n_samples=n, impurity=1.0, value=0.0, is_leaf=False,
        feature=feature, threshold=threshold, left=left, right=right,
    )


def hand_tree(sub):
    """Root asks x0 <= 0.5. The right side asks x1 <= 2.0."""
    inner = split(sub, 1, 2.0, leaf(sub, 3.0), leaf(sub, 5.0))
    return split(sub, 0, 0.5, leaf(sub, -1.0), inner)


def all_nodes(node):
    yield node
    if not node.is_leaf:
        yield from all_nodes(node.left)
        yield from all_nodes(node.right)


@check("predict_tree walks rows to leaves, and a row sitting on a threshold goes left")
def test_predict(sub):
    tree = hand_tree(sub)
    X = np.array([[0.0, 9.0], [0.5, 0.0], [1.0, 2.0], [1.0, 2.5], [10.0, -4.0]])
    close(sub.predict_tree(tree, X), np.array([-1.0, -1.0, 3.0, 5.0, 3.0]), atol=1e-12, label="leaf values")
    close(sub.predict_tree(leaf(sub, 7.0), X), np.full(5, 7.0), atol=1e-12, label="a tree that is one leaf")


@check("tree_depth counts splits on the longest path")
def test_depth(sub):
    equal(sub.tree_depth(leaf(sub, 1.0)), 0, label="a single leaf")
    equal(sub.tree_depth(split(sub, 0, 0.0, leaf(sub, 1.0), leaf(sub, 2.0))), 1, label="one split")
    equal(sub.tree_depth(hand_tree(sub)), 2, label="the hand-built tree")
    chain = leaf(sub, 0.0)
    for k in range(4):
        chain = split(sub, 0, float(k), leaf(sub, float(k)), chain)
    equal(sub.tree_depth(chain), 4, label="a lopsided chain of four splits")


@check("grow_tree on the eight houses cuts area at 4.75, then distance at 2.0, and ends with three leaves")
def test_grow_hand(sub):
    tree = sub.grow_tree(HOUSES, EXPENSIVE, criterion="gini")
    if tree.is_leaf:
        fail("the root is a leaf, but the eight houses can be split")
    equal(tree.feature, 0, label="root feature")
    almost(tree.threshold, 4.75, tol=1e-12, label="root threshold")
    equal(tree.n_samples, 8, label="rows at the root")
    almost(tree.impurity, 0.5, tol=1e-12, label="root impurity")
    left, right = tree.left, tree.right
    if not right.is_leaf:
        fail("the right box (areas above 4.75) holds only expensive houses, so it should be a leaf")
    equal(right.n_samples, 3, label="rows in the right leaf")
    almost(right.value, 1.0, tol=1e-12, label="right leaf value")
    if left.is_leaf:
        fail("the left box holds 5 mixed houses, so it should split again")
    equal(left.n_samples, 5, label="rows in the left box")
    equal(left.feature, 1, label="left box feature")
    almost(left.threshold, 2.0, tol=1e-12, label="left box threshold")
    close(
        np.array([left.left.value, left.right.value, left.left.n_samples, left.right.n_samples]),
        np.array([1.0, 0.0, 1, 4]),
        atol=1e-12,
        label="leaf values and sizes under the left box",
    )
    equal(sub.tree_depth(tree), 2, label="depth")
    equal(sub.count_leaves(tree), 3, label="leaves")


@check("every leaf predicts the mean of the training labels that reach it, for squared error and for Gini")
def test_leaf_is_mean(sub):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y_num = X[:, 0] ** 2 - X[:, 1] + 0.2 * rng.normal(size=200)
    y_cls = (X[:, 0] + 0.5 * X[:, 2] + 0.3 * rng.normal(size=200) > 0).astype(float)
    for criterion, y in (("mse", y_num), ("gini", y_cls)):
        tree = sub.grow_tree(X, y, criterion=criterion, max_depth=4, min_samples_leaf=5)
        pred = sub.predict_tree(tree, X)
        for value in np.unique(pred):
            mask = pred == value
            almost(value, float(y[mask].mean()), tol=1e-9, label=f"{criterion}: leaf {value:.5g} versus the mean of its {mask.sum()} rows")


@check("with no depth limit, every row gets its own leaf and the training error is zero")
def test_zero_error(sub):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    tree = sub.grow_tree(X, y, criterion="mse")
    close(sub.predict_tree(tree, X), y, atol=1e-9, rtol=1e-9, label="predictions on the training rows")
    equal(sub.count_leaves(tree), 60, label="leaves for 60 rows with 60 different labels")


@check("max_depth and min_samples_leaf are respected, and row counts add up")
def test_limits(sub):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(120, 2))
    y = np.sin(2 * X[:, 0]) + 0.3 * rng.normal(size=120)
    for depth in (0, 1, 2, 3):
        tree = sub.grow_tree(X, y, criterion="mse", max_depth=depth)
        if sub.tree_depth(tree) > depth:
            fail(f"max_depth={depth} but the tree has depth {sub.tree_depth(tree)}")
    root = sub.grow_tree(X, y, criterion="mse", max_depth=0)
    if not root.is_leaf:
        fail("max_depth=0 should give a single leaf")
    almost(root.value, float(y.mean()), tol=1e-12, label="the single leaf's value")
    tree = sub.grow_tree(X, y, criterion="mse", min_samples_leaf=7)
    for node in all_nodes(tree):
        if node.is_leaf and node.n_samples < 7:
            fail(f"a leaf holds {node.n_samples} rows, but min_samples_leaf is 7")
        if not node.is_leaf:
            equal(node.n_samples, node.left.n_samples + node.right.n_samples, label="rows in a node versus its two children")


@check("greedy search sees no gain in XOR, so the tree stays one leaf")
def test_xor(sub):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    for criterion in ("gini", "entropy"):
        tree = sub.grow_tree(X, y, criterion=criterion)
        if not tree.is_leaf:
            fail(f"under {criterion}, no single cut helps XOR, so the root should stay a leaf")
        almost(tree.value, 0.5, tol=1e-12, label="the leaf's class-1 fraction")
