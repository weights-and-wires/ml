"""Score a probability with the log loss.

You write three functions, in this order:

1. `softplus`: log(1 + exp(z)) without overflow.
2. `log_loss`: the loss the model minimises, with an optional ridge fee.
3. `naive_log_loss`: the textbook formula on probabilities, for comparison.

`sigmoid` is provided, because you wrote it in the previous chapter.

Shapes used everywhere: X is (n, d), one row per example; y is (n,) with
labels 0 or 1; w is (d,). The loss is a sum over examples, not a mean:

    L(w) = sum_i [softplus(z_i) - y_i * z_i] + (ridge / 2) * ||w||^2,
    where z = X @ w.
"""

import numpy as np


# Provided: you wrote this in the chapter "From a score to a probability".
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softplus(z: np.ndarray) -> np.ndarray:
    """Compute log(1 + exp(z)) elementwise, stable for any finite input.

    Args:
        z: Array of any shape.

    Returns:
        An array of the same shape. At z = 800 it returns 800, not infinity.

    Steps:
        1. Keep the larger of the two terms, max(z, 0).
        2. Add log(1 + exp(-|z|)), whose exponent is never positive.
    """
    raise NotImplementedError("write softplus")


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    """Return the log loss of the weights `w` on the data, as one float.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).
        y: Labels of 0 or 1, shape (n,).
        ridge: Strength of the ridge fee.

    Returns:
        sum(softplus(z) - y * z) + (ridge / 2) * ||w||^2, with z = X @ w.
        The loss must stay finite when a score is as large as 800.
    """
    raise NotImplementedError("write log_loss")


def naive_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Return the textbook log loss from probabilities, as one float.

    Args:
        p: Probabilities of the positive class, shape (n,).
        y: Labels of 0 or 1, shape (n,).

    Returns:
        -sum(y * log(p) + (1 - y) * log(1 - p)). It is allowed to break when
        a probability rounds to exactly 0 or 1; that is the point of the lesson.
    """
    raise NotImplementedError("write naive_log_loss")


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    y = np.array([1.0, 0.0])
    w = np.array([0.5, -1.0])
    print("log_loss       :", round(log_loss(w, X, y), 4))
    print("naive_log_loss :", round(naive_log_loss(sigmoid(X @ w), y), 4))
    print("loss at w = 0  :", round(log_loss(np.zeros(2), X, y), 4), "= 2 ln 2 =", round(2 * np.log(2), 4))
    # A score of 40 on a non-spam email: the textbook formula breaks, the stable one does not.
    z40 = np.array([[40.0]])
    with np.errstate(divide="ignore"):
        print("textbook at z = 40:", naive_log_loss(sigmoid(z40[:, 0]), np.array([0.0])))
    print("stable at z = 40  :", log_loss(np.array([1.0]), z40, np.array([0.0])))
