"""Impurity: how bad is one answer for a box of labels?

Fill in `gini`, `entropy`, and `variance_impurity`, in that order. `leaf_value`
is written for you. A box is a group of rows that a tree has not separated yet.
Labels are non-negative whole numbers for `gini` and `entropy`, and any numbers
for `variance_impurity`.
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
