"""Reference solution for the sigmoid and probability exercise."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


if __name__ == "__main__":
    # The two emails from the lesson: a bias column and the number of replies.
    X = np.array([[1.0, 0.0], [1.0, 2.0]])
    w = np.array([0.5, -1.0])
    p = predict_proba(w, X)
    print("scores       :", X @ w)
    print("probabilities:", np.round(p, 4))
    print("round trip   :", np.round(logit(p), 4))
    print("sigmoid at -800, 0, 800:", sigmoid(np.array([-800.0, 0.0, 800.0])))
