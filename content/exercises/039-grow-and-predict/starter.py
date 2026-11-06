"""Grow a tree: recurse on the best split, stop sensibly, and predict.

Fill in `predict_tree`, `tree_depth`, and `grow_tree`, in that order. The
impurity functions and `best_split` from the previous chapters are written for
you below, along with the `Node` class and `count_leaves`.
Labels are numbers for "mse", and 0 or 1 for "gini" and "entropy".
"""

from dataclasses import dataclass

import numpy as np

# --- Written in the chapter "Boxes, leaves, and impurity". Given here, already working. ---


def gini(y: np.ndarray) -> float:
    """Gini impurity of a label array: the chance two random draws disagree.

    Args:
        y: Class labels as non-negative whole numbers, shape (n,). Float labels
            such as 0.0 and 1.0 are allowed.

    Returns:
        1 - sum_k p_k^2 as a Python float, where p_k is the fraction of labels
        equal to class k. An empty array returns 0.0.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    p = counts / y.size
    return float(1.0 - np.sum(p**2))


def entropy(y: np.ndarray) -> float:
    """Entropy of a label array, in nats (natural log).

    Args:
        y: Class labels as non-negative whole numbers, shape (n,).

    Returns:
        -sum_k p_k log(p_k) as a Python float. A class with no labels adds
        nothing, so 0 log 0 counts as 0 and the result is never nan. An empty
        array returns 0.0.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    p = counts[counts > 0] / y.size
    return float(-np.sum(p * np.log(p)))


def variance_impurity(y: np.ndarray) -> float:
    """Variance impurity of numeric labels: the mean squared gap to the mean.

    Args:
        y: Numeric labels, shape (n,).

    Returns:
        mean((y - mean(y))^2) as a Python float. An empty or constant array
        returns 0.0.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    return float(np.mean((y - y.mean()) ** 2))


def leaf_value(y: np.ndarray) -> float:
    """The answer a leaf gives: the mean of the labels in its box.

    For 0/1 labels the mean is the fraction of ones. This function is written
    for you, because it is one line. The checks use it to confirm that the mean
    is the best single answer.
    """
    return float(np.mean(y))


def node_impurity(y: np.ndarray, criterion: str) -> float:
    """Impurity of a box under "mse" (variance), "gini", or "entropy"."""
    y = np.asarray(y)
    if criterion == "mse":
        return variance_impurity(y)
    if criterion == "gini":
        return gini(y)
    if criterion == "entropy":
        return entropy(y)
    raise ValueError(f"unknown criterion {criterion!r}")


# --- Written in the chapter "Which question helps most". Given here, already working. ---


def candidate_thresholds(x: np.ndarray, min_samples_leaf: int = 1) -> np.ndarray:
    """List every threshold worth trying on one feature.

    Sort the values and take the midpoint between each pair of consecutive
    values that differ. Equal neighbours give no threshold, because no cut can
    pull equal values apart. Keep a midpoint only when a row goes left exactly
    when x <= threshold, and both sides then hold at least `min_samples_leaf`
    rows.

    Args:
        x: One feature, shape (n,), in any order.
        min_samples_leaf: Fewest rows allowed on each side.

    Returns:
        The kept midpoints in increasing order, shape (k,). Empty when no cut is
        possible.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    n_left = np.arange(1, n)  # a cut after sorted row i leaves i + 1 rows on the left
    valid = xs[:-1] != xs[1:]
    valid &= (n_left >= min_samples_leaf) & (n - n_left >= min_samples_leaf)
    return (0.5 * (xs[:-1] + xs[1:]))[valid]


def _binary_impurity(p: np.ndarray, criterion: str) -> np.ndarray:
    """Impurity of a 0/1 box from its fraction of ones p, for many boxes at once."""
    if criterion == "gini":
        return 2.0 * p * (1.0 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return np.where((p > 0) & (p < 1), h, 0.0)


def best_split(
    X: np.ndarray, y: np.ndarray, criterion: str, min_samples_leaf: int = 1
) -> tuple[int, float, float] | None:
    """Find the split with the largest gain, scanning each feature in one sorted pass.

    Args:
        X: Feature matrix, shape (n, d).
        y: Labels, shape (n,). Numbers for "mse", 0/1 values for "gini" and "entropy".
        criterion: "mse", "gini", or "entropy".
        min_samples_leaf: Fewest rows allowed on each side of a split.

    Returns:
        (feature, threshold, gain) for the best split, or None when no feature
        offers a valid cut. A row goes left when X[row, feature] <= threshold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    if n < 2 * min_samples_leaf:
        return None

    parent = node_impurity(y, criterion)
    # Variance does not change when every label shifts, so centre first. That
    # keeps Q/n - (S/n)^2 from subtracting two huge, nearly equal numbers.
    y_run = y - y.mean() if criterion == "mse" else y

    best = None  # (gain, feature, threshold)
    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y_run[order]

        thresholds = candidate_thresholds(xs, min_samples_leaf)
        if thresholds.size == 0:
            continue
        n_left = np.searchsorted(xs, thresholds, side="right")
        n_right = n - n_left

        cum = np.cumsum(ys)
        left_sum = cum[n_left - 1]
        right_sum = cum[-1] - left_sum
        if criterion == "mse":
            cum_sq = np.cumsum(ys**2)
            left_sq = cum_sq[n_left - 1]
            right_sq = cum_sq[-1] - left_sq
            left_imp = left_sq / n_left - (left_sum / n_left) ** 2
            right_imp = right_sq / n_right - (right_sum / n_right) ** 2
        else:
            left_imp = _binary_impurity(left_sum / n_left, criterion)
            right_imp = _binary_impurity(right_sum / n_right, criterion)

        gain = parent - (n_left * left_imp + n_right * right_imp) / n
        k = int(np.argmax(gain))
        if best is None or gain[k] > best[0]:
            best = (float(gain[k]), j, float(thresholds[k]))

    if best is None:
        return None
    gain, feature, threshold = best
    return feature, threshold, gain


# --- A plain data holder, given. ---


@dataclass
class Node:
    """One node of a fitted tree.

    An internal node (is_leaf False) holds `feature`, `threshold`, `left`, and
    `right`; a row goes left when row[feature] <= threshold. Every node keeps
    the number of training rows that reached it, their impurity, and their
    mean label, which is the answer a leaf gives.
    """

    n_samples: int
    impurity: float
    value: float
    is_leaf: bool
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


def count_leaves(node: Node) -> int:
    """Number of leaves under this node. Written for you: it is the same
    recursion as `tree_depth`, with a sum in place of a max."""
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

# --- Your work in this chapter starts here. ---


def predict_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    """Walk every row from the root down to a leaf and return that leaf's value.

    Args:
        tree: Root Node of a tree.
        X: Rows to predict, shape (n, d).

    Returns:
        Predictions, shape (n,). At an internal node a row goes left when
        row[feature] <= threshold, and right otherwise.
    """
    raise NotImplementedError("predict_tree")


def tree_depth(node: Node) -> int:
    """Number of splits on the longest path from this node to a leaf.

    Args:
        node: Root of a tree or subtree.

    Returns:
        0 for a single leaf, 1 for one split with two leaves, and so on.
    """
    raise NotImplementedError("tree_depth")


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "mse",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> Node:
    """Grow a tree: take the best split, then repeat inside each half.

    Every node, leaf or not, records n_samples, impurity (under `criterion`),
    and value (the mean of its labels). Stop and keep a leaf when any of these
    holds: the box is pure (impurity <= 1e-12), the node sits at `max_depth`,
    `best_split` returns None, or the best gain is not above 1e-12. Send the
    rows with X[:, feature] <= threshold to the left child.

    Args:
        X: Feature matrix, shape (n, d).
        y: Labels, shape (n,). Numbers for "mse", 0/1 values for "gini" and "entropy".
        criterion: "mse", "gini", or "entropy".
        max_depth: Most splits allowed on any path from the root. None means no limit.
        min_samples_leaf: Fewest rows allowed in a leaf.

    Returns:
        The root Node.
    """
    raise NotImplementedError("grow_tree")

if __name__ == "__main__":
    # Eight houses: [area in tens of square metres, kilometres to the metro].
    houses = np.array(
        [[1.0, 5.0], [2.0, 3.0], [2.5, 1.0], [3.0, 6.0], [4.0, 4.0], [5.5, 2.0], [6.0, 5.0], [7.0, 1.0]]
    )
    expensive = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    tree = grow_tree(houses, expensive, criterion="gini")
    print(f"depth {tree_depth(tree)}, leaves {count_leaves(tree)}")
    print(f"root asks: feature {tree.feature} <= {tree.threshold}")
    new = np.array([[3.5, 1.5], [3.5, 4.0], [6.5, 3.0]])
    print(f"new houses {new.tolist()} -> {predict_tree(tree, new).tolist()}")

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 6, size=(40, 1))
    y = np.sin(x[:, 0]) + 0.3 * rng.normal(size=40)
    for depth in (1, 3, None):
        t = grow_tree(x, y, criterion="mse", max_depth=depth)
        err = np.mean((predict_tree(t, x) - y) ** 2)
        print(f"max_depth={depth}: {count_leaves(t)} leaves, training error {err:.4f}")
