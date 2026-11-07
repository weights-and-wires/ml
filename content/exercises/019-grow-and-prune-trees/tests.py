"""Checks for the grow and prune exercise."""

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

FOUR_X = np.array([[1.0], [2.0], [3.0], [4.0]])

FOUR_Y = np.array([1.0, 3.0, 8.0, 10.0])

def four_house_tree(sub):
    return sub.grow_tree(FOUR_X, FOUR_Y, criterion="mse")

def shape_of(node):
    """The whole tree as nested tuples, to compare before and after."""
    if node.is_leaf:
        return ("leaf", node.n_samples, node.impurity, node.value)
    return ("split", node.feature, node.threshold, node.n_samples, node.impurity, node.value, shape_of(node.left), shape_of(node.right))

def split_paths(node, path=""):
    """The set of (path, feature, threshold) for every internal node."""
    if node.is_leaf:
        return set()
    here = {(path, node.feature, node.threshold)}
    return here | split_paths(node.left, path + "L") | split_paths(node.right, path + "R")

def all_subtree_costs(node, alpha):
    """The cost of every tree you can get by collapsing some splits of `node`."""
    leaf_cost = node.n_samples * node.impurity + alpha
    if node.is_leaf:
        return [leaf_cost]
    costs = [leaf_cost]
    for a in all_subtree_costs(node.left, alpha):
        for b in all_subtree_costs(node.right, alpha):
            costs.append(a + b)
    return costs


def _part_zero_error(sub):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    tree = sub.grow_tree(X, y, criterion="mse")
    close(sub.predict_tree(tree, X), y, atol=1e-9, rtol=1e-9, label="predictions on the training rows")
    equal(sub.count_leaves(tree), 60, label="leaves for 60 rows with 60 different labels")


def _part_leaf_is_mean(sub):
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


def _part_cost_hand(sub):
    full = four_house_tree(sub)
    almost(sub.tree_cost(full, 0.0), 0.0, tol=1e-12, label="four leaves, alpha 0")
    almost(sub.tree_cost(full, 1.5), 6.0, tol=1e-12, label="four leaves, alpha 1.5")
    two = sub.Node(
        n_samples=4, impurity=13.25, value=5.5, is_leaf=False, feature=0, threshold=2.5,
        left=sub.Node(n_samples=2, impurity=1.0, value=2.0, is_leaf=True),
        right=sub.Node(n_samples=2, impurity=1.0, value=9.0, is_leaf=True),
    )
    almost(sub.tree_cost(two, 10.0), 24.0, tol=1e-12, label="two leaves, alpha 10")
    one = sub.Node(n_samples=4, impurity=13.25, value=5.5, is_leaf=True)
    almost(sub.tree_cost(one, 10.0), 63.0, tol=1e-12, label="one leaf, alpha 10")


def _part_cost_math(sub):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=100)
    tree = sub.grow_tree(X, y, criterion="mse", max_depth=3)
    sse = float(np.sum((y - sub.predict_tree(tree, X)) ** 2))
    almost(sub.tree_cost(tree, 0.0), sse, tol=1e-9, label="cost at alpha 0 versus the squared misses")
    leaves = sub.count_leaves(tree)
    almost(sub.tree_cost(tree, 2.0) - sub.tree_cost(tree, 1.0), float(leaves), tol=1e-9, label=f"cost per unit of alpha, {leaves} leaves")


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


@check("with no depth limit every row gets its own leaf and the training error is zero, and every leaf predicts the mean of its rows")
def test_zero_error_and_leaf_is_mean(sub):
    _part_zero_error(sub)
    _part_leaf_is_mean(sub)


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


@check("tree_cost adds training error and a fee per leaf: 4 + 2 alpha for the two-leaf tree, 53 + alpha for one leaf")
def test_cost_hand_and_cost_math(sub):
    _part_cost_hand(sub)
    _part_cost_math(sub)


@check("break_even_alpha is 49 for the four-house split, 2 for a pair of houses, and infinite for a leaf")
def test_break_even(sub):
    full = four_house_tree(sub)
    pruned = sub.Node(
        n_samples=4, impurity=13.25, value=5.5, is_leaf=False, feature=0, threshold=2.5,
        left=sub.Node(n_samples=2, impurity=1.0, value=2.0, is_leaf=True),
        right=sub.Node(n_samples=2, impurity=1.0, value=9.0, is_leaf=True),
    )
    almost(sub.break_even_alpha(pruned), 49.0, tol=1e-9, label="the split of four houses into two pairs")
    almost(sub.break_even_alpha(full.left), 2.0, tol=1e-9, label="the split of one pair")
    almost(sub.break_even_alpha(full), 53.0 / 3.0, tol=1e-9, label="the whole four-leaf tree as it stands")
    if sub.break_even_alpha(full.left.left) != float("inf"):
        fail("a leaf has nothing to collapse, so its break-even alpha should be float('inf')")


@check("cost_complexity_prune keeps the split for alpha 48 and collapses at 49, and drops the weak splits first")
def test_prune_hand(sub):
    full = four_house_tree(sub)
    for alpha, leaves in ((0.0, 4), (1.9, 4), (2.0, 2), (2.1, 2), (48.0, 2), (49.0, 1), (50.0, 1), (1e6, 1)):
        got = sub.count_leaves(sub.cost_complexity_prune(full, alpha))
        equal(got, leaves, label=f"leaves at alpha {alpha}")
    pruned = sub.cost_complexity_prune(full, 3.0)
    close(sub.predict_tree(pruned, FOUR_X), np.array([2.0, 2.0, 9.0, 9.0]), atol=1e-12, label="predictions of the two-leaf tree")


@check("cost_complexity_prune returns new nodes and leaves the input tree alone")
def test_no_mutation(sub):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] + 0.3 * rng.normal(size=80)
    full = sub.grow_tree(X, y, criterion="mse", max_depth=5)
    before = shape_of(full)
    originals = {id(node) for node in all_nodes(full)}
    for alpha in (0.0, 0.5, 5.0, 1e6):
        pruned = sub.cost_complexity_prune(full, alpha)
        if shape_of(full) != before:
            fail(f"the input tree changed after pruning at alpha {alpha}")
        if any(id(node) in originals for node in all_nodes(pruned)):
            fail(f"the pruned tree at alpha {alpha} shares a Node object with the input; build new nodes")


@check("the pruned tree is the cheapest of all trees reachable by collapsing, and larger alpha only removes splits")
def test_optimal_and_nested(sub):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 2))
    y = np.sin(2 * X[:, 0]) + 0.4 * rng.normal(size=60)
    full = sub.grow_tree(X, y, criterion="mse", max_depth=3)
    previous = None
    for alpha in (0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        pruned = sub.cost_complexity_prune(full, alpha)
        best = min(all_subtree_costs(full, alpha))
        almost(sub.tree_cost(pruned, alpha), best, tol=1e-9, label=f"cost at alpha {alpha} versus the cheapest possible subtree")
        paths = split_paths(pruned)
        if previous is not None and not paths <= previous:
            fail(f"raising alpha to {alpha} kept a split that was already gone at the smaller alpha")
        previous = paths
