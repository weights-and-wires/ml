"""Turn scores into probabilities and score them with the log loss.

You write six functions, in this order:

1. `sigmoid`: turn scores into probabilities without overflow.
2. `logit`: the inverse of the sigmoid, from a probability back to a score.
3. `predict_proba`: score a whole data matrix and return one probability per row.
4. `softplus`: log(1 + exp(z)) without overflow.
5. `naive_log_loss`: the textbook formula on probabilities, for comparison.
6. `log_loss`: the loss the model minimises, with an optional ridge fee.

Shapes used everywhere: X is (n, d), one row per example; y is (n,) with
labels 0 or 1; w is (d,); a score is one number per row, z = X @ w. The loss
is a sum over examples, not a mean:

    L(w) = sum_i [softplus(z_i) - y_i * z_i] + (ridge / 2) * ||w||^2.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the logistic sigmoid elementwise, stable for any finite input.

    Args:
        z: Array of any shape.

    Returns:
        An array of the same shape holding 1 / (1 + exp(-z)), with no overflow
        or invalid-value warning, even when `z` holds entries as large in
        magnitude as 800.

    Steps:
        1. Convert `z` to a float array and make an empty output array of the
           same shape.
        2. Build a mask for the entries where z >= 0.
        3. Fill each part with the formula that only exponentiates a number
           that is zero or negative.
    """
    raise NotImplementedError("write sigmoid")


def logit(p: np.ndarray) -> np.ndarray:
    """Compute the log-odds of a probability, the inverse of `sigmoid`.

    Args:
        p: Array of probabilities strictly between 0 and 1.

    Returns:
        An array of the same shape holding log(p / (1 - p)), so that
        logit(sigmoid(z)) returns z.
    """
    raise NotImplementedError("write logit")


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return the probability of the positive class for every row of X.

    Args:
        w: Weights, shape (d,).
        X: Data matrix, shape (n, d).

    Returns:
        Probabilities, shape (n,), each equal to sigmoid of the row's score.
    """
    raise NotImplementedError("write predict_proba")


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


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    y = np.array([1.0, 0.0])
    w = np.array([0.5, -1.0])
    p = predict_proba(w, X)
    print("scores         :", X @ w)
    print("probabilities  :", np.round(p, 4))
    print("round trip     :", np.round(logit(p), 4))
    print("sigmoid at -800, 0, 800:", sigmoid(np.array([-800.0, 0.0, 800.0])))
    print("log_loss       :", round(log_loss(w, X, y), 4))
    print("naive_log_loss :", round(naive_log_loss(p, y), 4))
    print("loss at w = 0  :", round(log_loss(np.zeros(2), X, y), 4), "= 2 ln 2 =", round(2 * np.log(2), 4))
    # A score of 40 on a non-spam email: the textbook formula breaks, the stable one does not.
    z40 = np.array([[40.0]])
    with np.errstate(divide="ignore"):
        print("textbook at z = 40:", naive_log_loss(sigmoid(z40[:, 0]), np.array([0.0])))
    print("stable at z = 40  :", log_loss(np.array([1.0]), z40, np.array([0.0])))
