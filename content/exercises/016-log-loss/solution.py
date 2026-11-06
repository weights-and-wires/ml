"""Reference solution for the log loss exercise."""

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
    # log(1 + exp(z)) with the larger of the two terms {1, exp(z)} factored
    # out, so the remaining exponential never overflows.
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def log_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    z = X @ w
    penalty = 0.5 * ridge * float(np.dot(w, w))
    return float(np.sum(softplus(z) - y * z)) + penalty


def naive_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


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
