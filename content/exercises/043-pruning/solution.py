"""Prune a tree: keep a split only if it pays for its extra leaves.

Fill in `tree_cost`, `break_even_alpha`, and `cost_complexity_prune`, in that
order. Everything from the previous chapters is written for you below: the
impurity functions, `best_split`, `grow_tree`, `predict_tree`, and
`count_leaves`.
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


# --- Written in the chapter "Grow a tree". Given here, already working. ---


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


def predict_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    """Walk every row from the root down to a leaf and return that leaf's value.

    Args:
        tree: Root Node of a tree.
        X: Rows to predict, shape (n, d).

    Returns:
        Predictions, shape (n,). At an internal node a row goes left when
        row[feature] <= threshold, and right otherwise.
    """
    X = np.asarray(X, dtype=float)
    out = np.empty(X.shape[0])
    for i, row in enumerate(X):
        node = tree
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        out[i] = node.value
    return out


def tree_depth(node: Node) -> int:
    """Number of splits on the longest path from this node to a leaf.

    Args:
        node: Root of a tree or subtree.

    Returns:
        0 for a single leaf, 1 for one split with two leaves, and so on.
    """
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    def build(X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        impurity = node_impurity(y, criterion)
        node = Node(
            n_samples=len(y), impurity=impurity, value=leaf_value(y), is_leaf=True
        )
        if impurity <= 1e-12:  # pure: nothing left to separate
            return node
        if max_depth is not None and depth >= max_depth:
            return node
        split = best_split(X, y, criterion, min_samples_leaf)
        if split is None:  # too few rows, or no feature can separate them
            return node
        feature, threshold, gain = split
        if gain <= 1e-12:  # no question helps
            return node

        goes_left = X[:, feature] <= threshold
        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        node.left = build(X[goes_left], y[goes_left], depth + 1)
        node.right = build(X[~goes_left], y[~goes_left], depth + 1)
        return node

    return build(X, y, 0)

# --- Your work in this chapter starts here. ---


def tree_cost(tree: Node, alpha: float) -> float:
    """Price of a tree: training error plus a fee of alpha per leaf.

    Args:
        tree: Root Node.
        alpha: Fee per leaf, alpha >= 0.

    Returns:
        sum over leaves of n_samples * impurity, plus alpha times the number of
        leaves, as a Python float.
    """
    def leaf_error(node: Node) -> tuple[float, int]:
        if node.is_leaf:
            return node.n_samples * node.impurity, 1
        err_l, leaves_l = leaf_error(node.left)
        err_r, leaves_r = leaf_error(node.right)
        return err_l + err_r, leaves_l + leaves_r

    error, leaves = leaf_error(tree)
    return float(error + alpha * leaves)


def break_even_alpha(node: Node) -> float:
    """The fee per leaf at which collapsing this subtree to one leaf costs the same as keeping it.

    Keeping the subtree as it stands costs R + alpha * L, where R is the sum of
    n_samples * impurity over its leaves and L is its number of leaves.
    Collapsing costs n_samples * impurity + alpha. Setting the two equal and
    solving for alpha gives the answer.

    Args:
        node: Root of a subtree.

    Returns:
        The break-even alpha as a Python float. A leaf has nothing to collapse,
        so return float("inf").
    """
    if node.is_leaf:
        return float("inf")
    kept_error = tree_cost(node, 0.0)  # training error of the subtree as it stands
    leaves = count_leaves(node)
    return float((node.n_samples * node.impurity - kept_error) / (leaves - 1))


def cost_complexity_prune(tree: Node, alpha: float) -> Node:
    """Return the cheapest tree you can reach by collapsing splits, at fee alpha per leaf.

    Work bottom-up. Prune both children first, then compare the cost of the
    node as a single leaf (n_samples * impurity + alpha) with the summed cost
    of its two pruned children. Collapse the node when the leaf costs no more.
    Build new Node objects; the input tree must not change.

    Args:
        tree: Root Node of a grown tree.
        alpha: Fee per leaf, alpha >= 0.

    Returns:
        The root of the pruned tree.
    """
    def prune(node: Node) -> tuple[Node, float]:
        leaf = Node(
            n_samples=node.n_samples,
            impurity=node.impurity,
            value=node.value,
            is_leaf=True,
        )
        leaf_cost = node.n_samples * node.impurity + alpha
        if node.is_leaf:
            return leaf, leaf_cost

        left, left_cost = prune(node.left)  # judge the children first
        right, right_cost = prune(node.right)
        keep_cost = left_cost + right_cost
        if leaf_cost <= keep_cost:  # a tie collapses: fewer leaves for the same price
            return leaf, leaf_cost

        kept = Node(
            n_samples=node.n_samples,
            impurity=node.impurity,
            value=node.value,
            is_leaf=False,
            feature=node.feature,
            threshold=node.threshold,
            left=left,
            right=right,
        )
        return kept, keep_cost

    return prune(tree)[0]

if __name__ == "__main__":
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 8.0, 10.0])
    full = grow_tree(x, y, criterion="mse")
    print(f"full tree: {count_leaves(full)} leaves, cost at alpha 1 = {tree_cost(full, 1.0)}")
    for alpha in (1.0, 3.0, 48.0, 50.0):
        pruned = cost_complexity_prune(full, alpha)
        print(f"alpha {alpha}: leaves {count_leaves(pruned)}, cost {tree_cost(pruned, alpha)}")
    two = cost_complexity_prune(full, 3.0)
    print(f"break-even alpha of the two-leaf tree: {break_even_alpha(two)}")

    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 6, size=(40, 1))
    ys = np.sin(xs[:, 0]) + 0.3 * rng.normal(size=40)
    big = grow_tree(xs, ys, criterion="mse")
    for alpha in (0.0, 0.05, 0.2, 1.0, 5.0):
        print(f"noisy data, alpha {alpha}: {count_leaves(cost_complexity_prune(big, alpha))} leaves")
