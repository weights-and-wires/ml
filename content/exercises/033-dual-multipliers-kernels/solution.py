"""Reference solution: the Gram matrix and the dual, scored by hand."""

import numpy as np


def gram_matrix(X: np.ndarray, kind: str = "linear", gamma: float = 1.0) -> np.ndarray:
    G = X @ X.T
    if kind == "linear":
        return G
    if kind == "poly":
        return (1.0 + G) ** 2
    if kind == "rbf":
        sq = np.diag(G)
        dist2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0)
        return np.exp(-gamma * dist2)
    raise ValueError(f"unknown kernel {kind!r}")


def dual_scores(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    return (alpha * y) @ K + b


def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    v = alpha * y
    return float(alpha.sum() - 0.5 * v @ K @ v)


if __name__ == "__main__":
    # Two houses on a line: a circle at x = -1 and a square at x = +1.
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, -1.0])
    alpha = np.array([0.5, 0.5])
    K = gram_matrix(X)
    w = (alpha * y) @ X
    print("Gram matrix:\n", K)
    print("w from the pushes:", w)
    print("scores:", dual_scores(alpha, y, K, 0.0))
    print("dual objective:", dual_objective(alpha, y, K), "and half ||w||^2:", 0.5 * float(w @ w))
    for a in (0.25, 0.5, 0.75):
        print(f"  alpha = [{a}, {a}] -> D = {dual_objective(np.array([a, a]), y, K):.4f}")
