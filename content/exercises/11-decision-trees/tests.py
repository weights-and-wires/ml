"""Checks for the decision tree exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape


def _node_impurity(y: np.ndarray, criterion: str) -> float:
    """An independent reference impurity, not imported from the solution."""
    y = np.asarray(y)
    if criterion == "mse":
        return float(np.mean((y - y.mean()) ** 2)) if y.size else 0.0
    counts = np.bincount(y.astype(int))
    p = counts[counts > 0] / counts.sum()
    if criterion == "gini":
        return float(1.0 - np.sum(p**2))
    if criterion == "entropy":
        return float(-np.sum(p * np.log(p)))
    raise ValueError(criterion)


def _brute_force_split(X, y, criterion, min_samples_leaf=1):
    """O(n^2 d) reference split search: try every threshold directly."""
    n, d = X.shape
    parent = _node_impurity(y, criterion)
    best = None  # (gain, feature, threshold)
    for j in range(d):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            threshold = 0.5 * (a + b)
            left_mask = X[:, j] <= threshold
            n_left, n_right = int(left_mask.sum()), int((~left_mask).sum())
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            imp_l = _node_impurity(y[left_mask], criterion)
            imp_r = _node_impurity(y[~left_mask], criterion)
            gain = parent - (n_left * imp_l + n_right * imp_r) / n
            if best is None or gain > best[0]:
                best = (gain, j, threshold)
    return best


def _recompute_gain(X, y, criterion, feature, threshold, min_samples_leaf=1):
    n = len(y)
    parent = _node_impurity(y, criterion)
    left_mask = X[:, feature] <= threshold
    n_left, n_right = int(left_mask.sum()), int((~left_mask).sum())
    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return None
    imp_l = _node_impurity(y[left_mask], criterion)
    imp_r = _node_impurity(y[~left_mask], criterion)
    return parent - (n_left * imp_l + n_right * imp_r) / n


def count_leaves(node) -> int:
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


@check("gini matches its definition at a pure node, a uniform node, and a mixed node")
def test_gini(sub):
    got = has(sub, "gini")
    almost(got(np.array([2, 2, 2, 2])), 0.0, tol=1e-12, label="pure node")
    almost(got(np.array([0, 1, 2, 3])), 0.75, tol=1e-12, label="four-way uniform node")

    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.bincount(y) / y.size
    want = 1.0 - np.sum(p**2)
    almost(got(y), want, tol=1e-10, label="a mixed three-class node")


@check("entropy matches its definition at a pure node, a uniform node, and a mixed node")
def test_entropy(sub):
    got = has(sub, "entropy")
    almost(got(np.array([1, 1, 1, 1])), 0.0, tol=1e-12, label="pure node")
    almost(
        got(np.array([0, 1, 2, 3])), np.log(4.0), tol=1e-10, label="four-way uniform node"
    )

    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.bincount(y) / y.size
    want = -np.sum(p * np.log(p))
    almost(got(y), want, tol=1e-10, label="a mixed three-class node")


@check("predict_tree walks a hand-built tree to the correct leaf")
def test_predict_tree(sub):
    Node = has(sub, "Node")
    predict = has(sub, "predict_tree")

    left = Node(n_samples=1, impurity=0.0, value=-1.0, is_leaf=True)
    right = Node(n_samples=1, impurity=0.0, value=1.0, is_leaf=True)
    root = Node(
        n_samples=2,
        impurity=1.0,
        value=0.0,
        is_leaf=False,
        feature=0,
        threshold=0.5,
        left=left,
        right=right,
    )
    X = np.array([[0.0], [1.0], [0.5], [-3.0], [10.0]])
    want = np.array([-1.0, 1.0, -1.0, -1.0, 1.0])
    close(predict(root, X), want, atol=1e-12, label="leaf values")


@check("best_split finds an obviously separating threshold")
def test_best_split_obvious(sub):
    got = has(sub, "best_split")
    X = np.arange(6.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    result = got(X, y, "gini")
    if result is None:
        fail("best_split found no split on a perfectly separable feature")
    feature, threshold, gain = result
    if feature != 0:
        fail(f"expected the only feature (0), got feature {feature}")
    if not (2.0 < threshold < 3.0):
        fail(f"expected a threshold strictly between the two classes, got {threshold}")
    almost(gain, 0.5, tol=1e-9, label="gain of a perfect binary split")


@check("best_split agrees with an O(n^2) brute-force search under squared error")
def test_best_split_brute_mse(sub):
    got = has(sub, "best_split")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=80)

    result = got(X, y, "mse")
    if result is None:
        fail("best_split found no split, but one clearly exists")
    feature, threshold, gain = result

    brute = _brute_force_split(X, y, "mse")
    if brute is None:
        fail("brute-force search itself found no split; the test data is degenerate")
    brute_gain = brute[0]

    almost(gain, brute_gain, tol=1e-8 * max(1.0, abs(brute_gain)), label="best gain")
    recomputed = _recompute_gain(X, y, "mse", feature, threshold)
    if recomputed is None:
        fail(f"the returned split (feature {feature}, threshold {threshold}) violates min_samples_leaf")
    almost(
        recomputed,
        brute_gain,
        tol=1e-8 * max(1.0, abs(brute_gain)),
        label="gain recomputed from the returned (feature, threshold)",
    )


@check("best_split agrees with an O(n^2) brute-force search under Gini impurity")
def test_best_split_brute_gini(sub):
    got = has(sub, "best_split")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)

    result = got(X, y, "gini")
    if result is None:
        fail("best_split found no split, but one clearly exists")
    feature, threshold, gain = result

    brute = _brute_force_split(X, y, "gini")
    brute_gain = brute[0]

    almost(gain, brute_gain, tol=1e-8 * max(1.0, abs(brute_gain)), label="best gain")
    recomputed = _recompute_gain(X, y, "gini", feature, threshold)
    if recomputed is None:
        fail(f"the returned split (feature {feature}, threshold {threshold}) violates min_samples_leaf")
    almost(
        recomputed,
        brute_gain,
        tol=1e-8 * max(1.0, abs(brute_gain)),
        label="gain recomputed from the returned (feature, threshold)",
    )


@check("a tree grown without a depth limit reaches zero training error")
def test_zero_training_error(sub):
    grow = has(sub, "grow_tree")
    predict = has(sub, "predict_tree")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]

    tree = grow(X, y, criterion="mse", max_depth=None, min_samples_leaf=1)
    pred = predict(tree, X)
    close(pred, y, atol=1e-8, rtol=1e-8, label="predictions on the training set")


@check("every leaf predicts the mean of the training labels routed to it")
def test_leaf_is_mean(sub):
    grow = has(sub, "grow_tree")
    predict = has(sub, "predict_tree")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] ** 2 - X[:, 1] + 0.2 * rng.normal(size=200)

    tree = grow(X, y, criterion="mse", max_depth=4, min_samples_leaf=5)
    pred = predict(tree, X)

    for value in np.unique(pred):
        mask = pred == value
        empirical_mean = float(y[mask].mean())
        almost(
            value,
            empirical_mean,
            tol=1e-8,
            label=f"leaf value {value:.6g} versus the mean of its {mask.sum()} training points",
        )


@check("cost_complexity_prune shrinks the tree monotonically as alpha grows")
def test_pruning_monotone(sub):
    grow = has(sub, "grow_tree")
    prune = has(sub, "cost_complexity_prune")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(150, 3))
    y = np.maximum(X[:, 0], 0) - np.maximum(-X[:, 1], 0) + 0.1 * rng.normal(size=150)

    full = grow(X, y, criterion="mse", max_depth=8, min_samples_leaf=2)
    full_leaves = count_leaves(full)
    if full_leaves < 4:
        fail(f"the unpruned tree only has {full_leaves} leaves; the test data should grow a larger tree")

    alphas = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.5, 500.0]
    leaf_counts = []
    for alpha in alphas:
        pruned = prune(full, alpha)
        leaf_counts.append(count_leaves(pruned))

    for i in range(len(leaf_counts) - 1):
        if leaf_counts[i] < leaf_counts[i + 1]:
            fail(
                "leaf count must not increase as alpha grows, but it went from "
                f"{leaf_counts[i]} at alpha={alphas[i]} to {leaf_counts[i + 1]} "
                f"at alpha={alphas[i + 1]}: {leaf_counts}"
            )

    if leaf_counts[0] != full_leaves:
        fail(
            f"at alpha=0 pruning should not collapse any split that has positive "
            f"gain, but the leaf count went from {full_leaves} to {leaf_counts[0]}"
        )
    if leaf_counts[-1] != 1:
        fail(
            f"a large enough alpha should collapse the tree to its root, but "
            f"alpha={alphas[-1]} still leaves {leaf_counts[-1]} leaves"
        )
