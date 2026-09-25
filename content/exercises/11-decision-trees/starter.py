"""CART: impurity, split search, growth, prediction, and pruning.

A tree is a `Node`. An internal node holds `feature` and `threshold` and
routes a row to `left` when `row[feature] <= threshold`, else to `right`. A
leaf has `is_leaf = True` and a scalar `value`, the prediction for every row
that reaches it. `value` is always the mean of the training labels routed to
that node: the mean for squared error, and the class-1 proportion for a
binary 0/1 label under Gini or entropy, since the two coincide (see the
lesson).

Labels `y` are either continuous (criterion "mse") or 0/1 integers
(criterion "gini" or "entropy"). Fill in the five functions below. None of
them should re-sort or re-scan data unnecessarily: `best_split` in particular
must run in O(n log n) per feature, using a single sorted pass with running
sums, not an O(n) recomputation of the impurity at every threshold.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """One node of a fitted tree."""

    n_samples: int
    impurity: float
    value: float
    is_leaf: bool
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


def gini(y: np.ndarray) -> float:
    """Gini impurity of an integer label array.

    Args:
        y: Integer class labels, shape (n,), values in {0, ..., K-1}.

    Returns:
        1 - sum_k p_k^2, where p_k is the fraction of labels equal to k.
    """
    raise NotImplementedError("gini")


def entropy(y: np.ndarray) -> float:
    """Entropy of an integer label array, in nats.

    Args:
        y: Integer class labels, shape (n,), values in {0, ..., K-1}.

    Returns:
        -sum_k p_k log(p_k), with the convention 0 log 0 = 0.
    """
    raise NotImplementedError("entropy")


def best_split(
    X: np.ndarray, y: np.ndarray, criterion: str, min_samples_leaf: int = 1
) -> tuple[int, float, float] | None:
    """Find the axis-aligned split with the largest impurity reduction.

    For each feature, sort the column once and sweep the threshold from left
    to right, maintaining running sums so that each candidate threshold costs
    O(1) to evaluate. A candidate threshold sits halfway between two distinct,
    consecutive sorted values, and is only valid when it leaves at least
    `min_samples_leaf` rows on each side.

    Args:
        X: Design matrix, shape (n, d).
        y: Labels, shape (n,). Continuous for criterion "mse", integer 0/1
            for criterion "gini" or "entropy".
        criterion: One of "mse", "gini", "entropy".
        min_samples_leaf: Minimum number of rows a split must leave on each
            side.

    Returns:
        A tuple `(feature, threshold, gain)` for the best split found, where
        `gain` is the parent impurity minus the sample-weighted impurity of
        the two children. Returns `None` when no valid split exists (fewer
        than `2 * min_samples_leaf` rows, or every feature constant).
    """
    raise NotImplementedError("best_split")


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "mse",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> Node:
    """Grow a tree by recursive greedy splitting.

    Stop recursing into a node when it is already pure, when `max_depth` is
    reached, when fewer than `2 * min_samples_leaf` rows remain, or when
    `best_split` finds no split with strictly positive gain.

    Args:
        X: Design matrix, shape (n, d).
        y: Labels, shape (n,), matching the convention in `best_split`.
        criterion: One of "mse", "gini", "entropy".
        max_depth: Maximum number of splits from the root to a leaf. `None`
            means no limit.
        min_samples_leaf: Minimum number of rows in a leaf.

    Returns:
        The root `Node` of the fitted tree.
    """
    raise NotImplementedError("grow_tree")


def predict_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    """Predict by walking each row from the root to a leaf.

    Args:
        tree: A `Node` returned by `grow_tree` (or a pruned version of one).
        X: Query rows, shape (n, d).

    Returns:
        The leaf value reached by each row, shape (n,).
    """
    raise NotImplementedError("predict_tree")


def cost_complexity_prune(tree: Node, alpha: float) -> Node:
    """Return the subtree minimising R_alpha(T) = R(T) + alpha * #leaves.

    Work bottom-up. At a leaf, the cost of keeping it is
    `n_samples * impurity + alpha`. At an internal node, recursively prune
    both children, then compare the cost of collapsing the node to a leaf
    against the summed cost of keeping the (already pruned) children, and
    keep whichever is cheaper.

    Args:
        tree: The fully grown (or already pruned) root `Node`.
        alpha: Nonnegative complexity penalty per leaf.

    Returns:
        A new tree (the input is not mutated) that minimises R_alpha among
        every subtree of `tree` reachable by collapsing internal nodes.
    """
    raise NotImplementedError("cost_complexity_prune")


def _count_leaves(node: Node) -> int:
    if node.is_leaf:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(-3.0, 3.0, size=(n, 2))
    y_reg = np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.05 * rng.normal(size=n)

    tree = grow_tree(X, y_reg, criterion="mse", max_depth=4, min_samples_leaf=5)
    pred = predict_tree(tree, X)
    print(f"depth-4 regression tree: training MSE {np.mean((pred - y_reg) ** 2):.4f}")

    full = grow_tree(X, y_reg, criterion="mse")
    print(f"unpruned leaves: {_count_leaves(full)}")
    for alpha in (0.0, 0.001, 0.01, 0.1):
        pruned = cost_complexity_prune(full, alpha)
        print(f"alpha={alpha}: {_count_leaves(pruned)} leaves")

    y_clf = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf_tree = grow_tree(X, y_clf, criterion="gini", max_depth=3, min_samples_leaf=5)
    clf_pred = predict_tree(clf_tree, X)
    accuracy = np.mean((clf_pred > 0.5).astype(int) == y_clf)
    print(f"depth-3 classification tree: training accuracy {accuracy:.3f}")
