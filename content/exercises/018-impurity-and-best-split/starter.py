"""Impurity, gain, and the best split: how a tree picks its question.

Fill in six functions, in this order. `gini`, `entropy`, and `variance_impurity`
score how badly one answer fits a box of labels. `split_gain` scores a question
by the impurity it removes. `candidate_thresholds` lists the cuts worth trying
on one feature, and `best_split` scores them all in one sorted pass.
`leaf_value`, `node_impurity`, and a small helper are written for you.
Labels are non-negative whole numbers for `gini` and `entropy`, and any numbers
for `variance_impurity`. For `split_gain` and `best_split`, labels are numbers
for "mse", and 0 or 1 for "gini" and "entropy".
"""

import numpy as np


def gini(y: np.ndarray) -> float:
    """Gini impurity of a label array: the chance two random draws disagree.

    Args:
        y: Class labels as non-negative whole numbers, shape (n,). Float labels
            such as 0.0 and 1.0 are allowed.

    Returns:
        1 - sum_k p_k^2 as a Python float, where p_k is the fraction of labels
        equal to class k. An empty array returns 0.0.
    """
    raise NotImplementedError("gini")


def entropy(y: np.ndarray) -> float:
    """Entropy of a label array, in nats (natural log).

    Args:
        y: Class labels as non-negative whole numbers, shape (n,).

    Returns:
        -sum_k p_k log(p_k) as a Python float. A class with no labels adds
        nothing, so 0 log 0 counts as 0 and the result is never nan. An empty
        array returns 0.0.
    """
    raise NotImplementedError("entropy")


def variance_impurity(y: np.ndarray) -> float:
    """Variance impurity of numeric labels: the mean squared gap to the mean.

    Args:
        y: Numeric labels, shape (n,).

    Returns:
        mean((y - mean(y))^2) as a Python float. An empty or constant array
        returns 0.0.
    """
    raise NotImplementedError("variance_impurity")


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

def split_gain(y: np.ndarray, left_mask: np.ndarray, criterion: str) -> float:
    """Gain of one split: parent impurity minus the size-weighted child impurities.

    Args:
        y: Labels of the rows in the box, shape (n,).
        left_mask: Boolean array, shape (n,). True sends a row to the left child.
        criterion: "mse" (variance), "gini", or "entropy".

    Returns:
        I_parent - (n_left / n) * I_left - (n_right / n) * I_right as a Python
        float. When either child is empty the split does nothing, so return 0.0.
    """
    raise NotImplementedError("split_gain")


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
    raise NotImplementedError("candidate_thresholds")


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
    raise NotImplementedError("best_split")


if __name__ == "__main__":
    houses = np.array([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0])
    print(f"one box of eight: answer {leaf_value(houses):.2f}, impurity {variance_impurity(houses):.2f}")
    cheap, dear = houses[:4], houses[4:]
    print(
        f"two boxes of four: answers {leaf_value(cheap):.2f} and {leaf_value(dear):.2f}, "
        f"impurities {variance_impurity(cheap):.2f} and {variance_impurity(dear):.2f}"
    )
    labels = np.array([0, 0, 1, 1, 1, 2])
    print(f"labels {labels.tolist()}: gini {gini(labels):.4f}, entropy {entropy(labels):.4f}")
    ones = np.array([0, 0, 0, 1])
    print(f"0/1 box {ones.tolist()}: gini {gini(ones):.4f}, twice the variance {2 * variance_impurity(ones):.4f}")

    area = np.array([[1.0], [2.0], [3.0], [4.0]])
    price = np.array([1.0, 3.0, 8.0, 10.0])
    print(f"candidate cuts: {candidate_thresholds(area[:, 0]).tolist()}")
    for t in candidate_thresholds(area[:, 0]):
        gain = split_gain(price, area[:, 0] <= t, "mse")
        print(f"cut at {t}: gain {gain:.2f}")
    print(f"best split: {best_split(area, price, 'mse')}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = (X[:, 1] > 0.2).astype(int)
    feature, threshold, gain = best_split(X, y, "gini")
    print(f"1000 rows, a hidden rule on feature 1: found feature {feature}, threshold {threshold:.3f}, gain {gain:.3f}")
