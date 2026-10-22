"""The Gram matrix and the dual, scored by hand.

Fill in the three functions below. Labels are in {-1, +1}. `alpha[i]` is how hard
house i pushes on the road, and `K[i, j]` says how alike houses i and j are.
"""

import numpy as np


def gram_matrix(X: np.ndarray, kind: str = "linear", gamma: float = 1.0) -> np.ndarray:
    """Return the (n, n) table of pairwise similarities between the rows of X.

    Args:
        X: Points, shape (n, d).
        kind: "linear" gives x_i . x_j. "poly" gives (1 + x_i . x_j) ** 2.
            "rbf" gives exp(-gamma * ||x_i - x_j||^2).
        gamma: The RBF width. Ignored by the other kinds.

    Returns:
        A symmetric array of shape (n, n). The RBF diagonal is exactly 1.
    """
    raise NotImplementedError("gram_matrix")


def dual_scores(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    """Return the score f_i = sum_k alpha_k y_k K[i, k] + b of every training point.

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Offset.

    Returns:
        An array of shape (n,). With the linear K this equals X @ w + b, where
        w = (alpha * y) @ X.
    """
    raise NotImplementedError("dual_scores")


def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """Return D(alpha) = sum_i alpha_i - 0.5 * sum_ij alpha_i alpha_j y_i y_j K[i, j].

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).

    Returns:
        One float.
    """
    raise NotImplementedError("dual_objective")


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
