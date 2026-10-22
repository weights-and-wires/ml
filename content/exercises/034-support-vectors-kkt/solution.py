"""Reference solution: sort the houses and run the KKT checklist."""

import numpy as np


# Written in the chapter "Pushes from each point: the dual and kernels".
def dual_scores(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    return (alpha * y) @ K + b


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
