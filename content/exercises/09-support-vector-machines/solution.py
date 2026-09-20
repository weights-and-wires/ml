"""Reference solution for the support vector machine exercise."""

import numpy as np


def hinge_loss_and_subgradient(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, C: float):
    margins = y * (X @ w + b)
    loss = 0.5 * float(w @ w) + C * float(np.sum(np.maximum(0.0, 1.0 - margins)))
    active = margins < 1.0
    grad_w = w - C * (y[active] @ X[active])
    grad_b = -C * float(np.sum(y[active]))
    return loss, grad_w, grad_b


def kkt_violation(
    alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, tol: float = 1e-8
) -> float:
    f = (alpha * y) @ K + b
    g = y * f - 1.0
    at_lower = alpha <= tol
    at_upper = alpha >= C - tol
    interior = ~(at_lower | at_upper)

    violation = np.zeros_like(alpha)
    violation[at_lower] = np.maximum(0.0, -g[at_lower])
    violation[at_upper] = np.maximum(0.0, g[at_upper])
    violation[interior] = np.abs(g[interior])
    return float(violation.max()) if len(violation) else 0.0


def smo_step(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, i: int, j: int):
    alpha = alpha.copy()
    if i == j:
        return alpha, b

    ai_old, aj_old = alpha[i], alpha[j]
    f_i = float((alpha * y) @ K[i] + b)
    f_j = float((alpha * y) @ K[j] + b)
    E_i, E_j = f_i - y[i], f_j - y[j]

    eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
    if eta <= 1e-12:
        return alpha, b

    if y[i] != y[j]:
        lo = max(0.0, aj_old - ai_old)
        hi = min(C, C + aj_old - ai_old)
    else:
        lo = max(0.0, ai_old + aj_old - C)
        hi = min(C, ai_old + aj_old)
    if lo >= hi:
        return alpha, b

    aj_new = aj_old + y[j] * (E_i - E_j) / eta
    aj_new = min(max(aj_new, lo), hi)
    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
    alpha[i], alpha[j] = ai_new, aj_new

    b1 = b - E_i - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
    b2 = b - E_j - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]
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
        num_changed = 0
        for i in range(n):
            f_i = float((alpha * y) @ K[i] + b)
            E_i = f_i - y[i]
            violates = (y[i] * E_i < -tol and alpha[i] < C - 1e-12) or (
                y[i] * E_i > tol and alpha[i] > 1e-12
            )
            if not violates:
                continue
            for j in range(n):
                if j == i:
                    continue
                new_alpha, new_b = smo_step(alpha, y, K, b, C, i, j)
                if abs(new_alpha[j] - alpha[j]) > 1e-7:
                    alpha, b = new_alpha, new_b
                    num_changed += 1
                    break
        if kkt_violation(alpha, y, K, b, C) < tol:
            break
        if num_changed == 0:
            break
    return alpha, b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 16, 2
    mu0, mu1 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    X = np.vstack(
        [mu0 + 0.5 * rng.normal(size=(n // 2, d)), mu1 + 0.5 * rng.normal(size=(n // 2, d))]
    )
    y = np.concatenate([-np.ones(n // 2), np.ones(n // 2)])
    K = X @ X.T
    C = 1.0

    alpha, b = fit_smo(K, y, C)
    w = (alpha * y) @ X
    print(f"support vectors: {(alpha > 1e-6).sum()} of {n}")
    print(f"w = {w}, b = {b:.4f}")
    print(f"KKT violation at convergence: {kkt_violation(alpha, y, K, b, C):.2e}")

    loss, grad_w, grad_b = hinge_loss_and_subgradient(w, b, X, y, C)
    print(f"hinge objective at the SMO solution: {loss:.4f}")
