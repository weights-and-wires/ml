"""Checks for the pruning exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail

FOUR_X = np.array([[1.0], [2.0], [3.0], [4.0]])
FOUR_Y = np.array([1.0, 3.0, 8.0, 10.0])


def four_house_tree(sub):
    return sub.grow_tree(FOUR_X, FOUR_Y, criterion="mse")


def all_nodes(node):
    yield node
    if not node.is_leaf:
        yield from all_nodes(node.left)
        yield from all_nodes(node.right)


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


@check("tree_cost adds training error and a fee per leaf: 4 + 2 alpha for the two-leaf tree, 53 + alpha for one leaf")
def test_cost_hand(sub):
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


@check("at alpha 0 the training part of tree_cost is the sum of squared misses, and each leaf adds alpha")
def test_cost_math(sub):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=100)
    tree = sub.grow_tree(X, y, criterion="mse", max_depth=3)
    sse = float(np.sum((y - sub.predict_tree(tree, X)) ** 2))
    almost(sub.tree_cost(tree, 0.0), sse, tol=1e-9, label="cost at alpha 0 versus the squared misses")
    leaves = sub.count_leaves(tree)
    almost(sub.tree_cost(tree, 2.0) - sub.tree_cost(tree, 1.0), float(leaves), tol=1e-9, label=f"cost per unit of alpha, {leaves} leaves")


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


@check("leaf counts never rise as alpha grows, alpha 0 keeps every useful split, and a huge alpha leaves the root")
def test_monotone(sub):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(150, 3))
    y = np.maximum(X[:, 0], 0) - np.maximum(-X[:, 1], 0) + 0.1 * rng.normal(size=150)
    full = sub.grow_tree(X, y, criterion="mse", max_depth=8, min_samples_leaf=2)
    total = sub.count_leaves(full)
    if total < 4:
        fail(f"the unpruned tree has only {total} leaves; the test needs a larger one")
    alphas = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.5, 5.0, 500.0]
    counts = [sub.count_leaves(sub.cost_complexity_prune(full, a)) for a in alphas]
    for k in range(len(counts) - 1):
        if counts[k] < counts[k + 1]:
            fail(f"leaf count rose from {counts[k]} at alpha {alphas[k]} to {counts[k + 1]} at alpha {alphas[k + 1]}: {counts}")
    equal(counts[0], total, label="leaves at alpha 0")
    equal(counts[-1], 1, label="leaves at alpha 500")


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
