"""Reference solution for the decision tree exercise."""

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
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


def entropy(y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _binary_gini(p: np.ndarray) -> np.ndarray:
    return 2.0 * p * (1.0 - p)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p)
    for q in (p, 1.0 - p):
        mask = q > 0
        out = out - np.where(mask, q * np.log(np.where(mask, q, 1.0)), 0.0)
    return out


def _node_impurity(y: np.ndarray, criterion: str) -> float:
    if criterion == "mse":
        return float(np.mean((y - y.mean()) ** 2)) if y.size else 0.0
    if criterion == "gini":
        return gini(y.astype(int))
    if criterion == "entropy":
        return entropy(y.astype(int))
    raise ValueError(f"unknown criterion {criterion!r}")


def best_split(
    X: np.ndarray, y: np.ndarray, criterion: str, min_samples_leaf: int = 1
) -> tuple[int, float, float] | None:
    n, d = X.shape
    if n < 2 * min_samples_leaf:
        return None

    parent_impurity = _node_impurity(y, criterion)
    best = None  # (gain, feature, threshold)

    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y[order]

        # Candidate split positions: after index i (0-indexed), splitting the
        # sorted row into ys[:i+1] (left) and ys[i+1:] (right).
        i = np.arange(n - 1)
        valid = xs[:-1] != xs[1:]
        n_left = i + 1
        n_right = n - n_left
        valid &= (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not np.any(valid):
            continue

        if criterion == "mse":
            cum_sum = np.cumsum(ys)
            cum_sq = np.cumsum(ys**2)
            total_sum, total_sq = cum_sum[-1], cum_sq[-1]
            left_sum = cum_sum[:-1]
            left_sq = cum_sq[:-1]
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq
            left_imp = left_sq / n_left - (left_sum / n_left) ** 2
            right_imp = right_sq / n_right - (right_sum / n_right) ** 2
        else:
            cum_pos = np.cumsum(ys)
            total_pos = cum_pos[-1]
            left_pos = cum_pos[:-1]
            right_pos = total_pos - left_pos
            p_left = left_pos / n_left
            p_right = right_pos / n_right
            fn = _binary_gini if criterion == "gini" else _binary_entropy
            left_imp = fn(p_left)
            right_imp = fn(p_right)

        weighted = (n_left * left_imp + n_right * right_imp) / n
        gain = parent_impurity - weighted
        gain = np.where(valid, gain, -np.inf)

        best_i = int(np.argmax(gain))
        if gain[best_i] == -np.inf:
            continue
        threshold = 0.5 * (xs[best_i] + xs[best_i + 1])
        candidate = (float(gain[best_i]), j, float(threshold))
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None
    g, feature, threshold = best
    return feature, threshold, g


def grow_tree(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "mse",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
) -> Node:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    def build(X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        impurity = _node_impurity(y, criterion)
        value = float(y.mean())
        node = Node(n_samples=len(y), impurity=impurity, value=value, is_leaf=True)

        if impurity <= 1e-12:
            return node
        if max_depth is not None and depth >= max_depth:
            return node
        split = best_split(X, y, criterion, min_samples_leaf)
        if split is None:
            return node
        feature, threshold, gain = split
        if gain <= 1e-12:
            return node

        mask = X[:, feature] <= threshold
        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        node.left = build(X[mask], y[mask], depth + 1)
        node.right = build(X[~mask], y[~mask], depth + 1)
        return node

    return build(X, y, 0)


def predict_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        node = tree
        while not node.is_leaf:
            node = node.left if X[i, node.feature] <= node.threshold else node.right
        out[i] = node.value
    return out


def cost_complexity_prune(tree: Node, alpha: float) -> Node:
    def prune(node: Node) -> tuple[Node, float]:
        leaf_cost = node.n_samples * node.impurity + alpha
        if node.is_leaf:
            return Node(
                n_samples=node.n_samples,
                impurity=node.impurity,
                value=node.value,
                is_leaf=True,
            ), leaf_cost

        left, left_cost = prune(node.left)
        right, right_cost = prune(node.right)
        subtree_cost = left_cost + right_cost

        if leaf_cost <= subtree_cost:
            return Node(
                n_samples=node.n_samples,
                impurity=node.impurity,
                value=node.value,
                is_leaf=True,
            ), leaf_cost

        new_node = Node(
            n_samples=node.n_samples,
            impurity=node.impurity,
            value=node.value,
            is_leaf=False,
            feature=node.feature,
            threshold=node.threshold,
            left=left,
            right=right,
        )
        return new_node, subtree_cost

    pruned, _ = prune(tree)
    return pruned


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
