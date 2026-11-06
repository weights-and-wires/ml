"""Reference solution: the SMO solver, two pushes at a time."""

import numpy as np


# Written in the chapter "The dual problem and the support vectors".
def dual_scores(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    return (alpha * y) @ K + b


# Written in the chapter "The dual problem and the support vectors".
def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    v = alpha * y
    return float(alpha.sum() - 0.5 * v @ K @ v)


# Written in the chapter "The dual problem and the support vectors".
def margin_groups(alpha: np.ndarray, C: float, tol: float = 1e-8) -> np.ndarray:
    groups = np.ones(len(alpha), dtype=int)
    groups[alpha <= tol] = 0
    groups[alpha >= C - tol] = 2
    return groups


# Written in the chapter "The dual problem and the support vectors".
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


def clip_bounds(alpha: np.ndarray, y: np.ndarray, C: float, i: int, j: int):
    if y[i] != y[j]:
        low = max(0.0, alpha[j] - alpha[i])
        high = min(C, C + alpha[j] - alpha[i])
    else:
        low = max(0.0, alpha[i] + alpha[j] - C)
        high = min(C, alpha[i] + alpha[j])
    return float(low), float(high)


def smo_step(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, i: int, j: int):
    alpha = alpha.copy()
    if i == j:
        return alpha, b

    f = dual_scores(alpha, y, K, b)
    E_i, E_j = f[i] - y[i], f[j] - y[j]
    eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
    if eta <= 1e-12:
        return alpha, b
    low, high = clip_bounds(alpha, y, C, i, j)
    if low >= high:
        return alpha, b

    ai_old, aj_old = alpha[i], alpha[j]
    aj_new = np.clip(aj_old + y[j] * (E_i - E_j) / eta, low, high)
    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
    alpha[i], alpha[j] = ai_new, aj_new

    di, dj = ai_new - ai_old, aj_new - aj_old
    b1 = b - E_i - y[i] * di * K[i, i] - y[j] * dj * K[i, j]
    b2 = b - E_j - y[i] * di * K[i, j] - y[j] * dj * K[j, j]
    if 0.0 < ai_new < C:
        b_new = b1
    elif 0.0 < aj_new < C:
        b_new = b2
    else:
        b_new = 0.5 * (b1 + b2)
    return alpha, float(b_new)


def fit_smo(K: np.ndarray, y: np.ndarray, C: float, max_iter: int = 200, tol: float = 1e-3):
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    for _ in range(max_iter):
        changed = 0
        for i in range(n):
            y_E = y[i] * (dual_scores(alpha, y, K, b)[i] - y[i])
            breaks = (y_E < -tol and alpha[i] < C) or (y_E > tol and alpha[i] > 0.0)
            if not breaks:
                continue
            for j in range(n):
                if j == i:
                    continue
                new_alpha, new_b = smo_step(alpha, y, K, b, C, i, j)
                if abs(new_alpha[j] - alpha[j]) > 1e-7:
                    alpha, b = new_alpha, new_b
                    changed += 1
                    break
        if kkt_violation(alpha, y, K, b, C) < tol or changed == 0:
            break
    return alpha, b


if __name__ == "__main__":
    # The two-house example: the pair (0, 1) is solved by a single step.
    X2 = np.array([[1.0], [-1.0]])
    y2 = np.array([1.0, -1.0])
    K2 = X2 @ X2.T
    a2, b2 = smo_step(np.zeros(2), y2, K2, 0.0, 10.0, 0, 1)
    print("one step on the pair:", a2, b2, "D =", dual_objective(a2, y2, K2))

    # Sixteen houses in two clusters.
    rng = np.random.default_rng(0)
    n, d = 16, 2
    X = np.vstack(
        [np.array([-2.0, 0.0]) + 0.5 * rng.normal(size=(8, d)), np.array([2.0, 0.0]) + 0.5 * rng.normal(size=(8, d))]
    )
    y = np.concatenate([-np.ones(8), np.ones(8)])
    K = X @ X.T
    C = 1.0
    alpha, b = fit_smo(K, y, C)
    w = (alpha * y) @ X
    print(f"support vectors: {(alpha > 1e-6).sum()} of {n}")
    print(f"w = {w}, b = {b:.4f}")
    print(f"KKT violation at the end: {kkt_violation(alpha, y, K, b, C):.2e}")
    print(f"dual objective: {dual_objective(alpha, y, K):.4f}, sum of alpha*y: {alpha @ y:.1e}")
