"""Turn scores into probabilities.

You write three functions, in this order:

1. `sigmoid`: turn scores into probabilities without overflow.
2. `logit`: the inverse of the sigmoid, from a probability back to a score.
3. `predict_proba`: score a whole data matrix and return one probability per row.

Shapes used everywhere: X is (n, d), one row per example, w is (d,), and a
score is one number per row, z = X @ w.
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


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    w = np.array([0.5, -1.0])
    p = predict_proba(w, X)
    print("scores       :", X @ w)
    print("probabilities:", np.round(p, 4))
    print("round trip   :", np.round(logit(p), 4))
    print("sigmoid at -800, 0, 800:", sigmoid(np.array([-800.0, 0.0, 800.0])))
