"""Gain and the best split: score every question about one feature, fast.

Fill in `split_gain`, `candidate_thresholds`, and `best_split`, in that order.
The impurity functions from the previous chapter are written for you below.
Labels are numbers for "mse", and 0 or 1 for "gini" and "entropy".
"""

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

# --- Your work in this chapter starts here. ---


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
    """Impurity of a 0/1 box from its fraction of ones p, for many boxes at once.

    Written for you: it works on a whole array of fractions, which is what the
    sweep produces, where `gini` and `entropy` take labels.
    """
    if criterion == "gini":
        return 2.0 * p * (1.0 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return np.where((p > 0) & (p < 1), h, 0.0)


def best_split(
    X: np.ndarray, y: np.ndarray, criterion: str, min_samples_leaf: int = 1
) -> tuple[int, float, float] | None:
    """Find the split with the largest gain, scanning each feature in one sorted pass.

    For each feature, sort once, get the thresholds from `candidate_thresholds`,
    and keep running sums so that each threshold costs one update. For "mse"
    keep the running sum S and sum of squares Q of the labels on the left, and
    use V = Q/n - (S/n)^2 on each side. For "gini" and "entropy" the labels are
    0 or 1, so a running count of ones gives the fraction p on each side.
    Centre y first for "mse", so that large labels do not wipe out the digits.

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
