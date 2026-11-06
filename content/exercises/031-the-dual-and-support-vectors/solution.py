"""Reference solution: the dual, the support vectors, and the KKT checklist."""

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


def margin_groups(alpha: np.ndarray, C: float, tol: float = 1e-8) -> np.ndarray:
    groups = np.ones(len(alpha), dtype=int)
    groups[alpha <= tol] = 0
    groups[alpha >= C - tol] = 2
    return groups


def slack_amounts(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    return np.maximum(0.0, 1.0 - y * dual_scores(alpha, y, K, b))


def kkt_violation(
    alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, tol: float = 1e-8
) -> float:
    g = y * dual_scores(alpha, y, K, b) - 1.0
    groups = margin_groups(alpha, C, tol)
    violation = np.where(
        groups == 0,
        np.maximum(0.0, -g),
        np.where(groups == 1, np.abs(g), np.maximum(0.0, g)),
    )
    return float(violation.max()) if len(violation) else 0.0


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

    # Six houses on a line, and the best pushes for C = 1.
    x = np.array([-3.0, -1.0, 0.5, 1.0, 3.0, 2.0])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    K = np.outer(x, x)
    C = 1.0
    alpha = np.array([0.0, 1 / 18, 1.0, 1.0, 0.0, 1 / 18])
    b = -1 / 3
    print("groups:", margin_groups(alpha, C))
    print("support vectors:", int((alpha > 1e-8).sum()), "of", len(alpha))
    print("slack:", np.round(slack_amounts(alpha, y, K, b), 4))
    print("violation at the best pushes:", kkt_violation(alpha, y, K, b, C))
    print("violation at alpha = 0, b = 0:", kkt_violation(np.zeros(6), y, K, 0.0, C))
