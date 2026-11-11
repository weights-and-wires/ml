"""Reference solution for the convex duality exercise."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def lagrangian(
    x: np.ndarray,
    lam: Sequence[float],
    nu: Sequence[float],
    f: Callable,
    gs: Sequence[Callable],
    hs: Sequence[Callable],
) -> float:
    total = float(f(x))
    total += sum(float(l) * float(g(x)) for l, g in zip(lam, gs))
    total += sum(float(n) * float(h(x)) for n, h in zip(nu, hs))
    return float(total)


def kkt_residuals(
    Q: np.ndarray, c: np.ndarray, G: np.ndarray, h: np.ndarray, x: np.ndarray, lam: np.ndarray
) -> dict:
    slack = G @ x - h
    return {
        "stationarity": float(np.max(np.abs(Q @ x + c + G.T @ lam))),
        "primal": float(max(0.0, np.max(slack))),
        "dual": float(max(0.0, np.max(-lam))),
        "complementary": float(np.max(np.abs(lam * slack))),
    }


def qp_dual_function(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, nu: np.ndarray) -> float:
    r = c + A.T @ nu
    return float(-0.5 * r @ np.linalg.solve(Q, r) - b @ nu)


def qp_dual(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    QinvA_T = np.linalg.solve(Q, A.T)
    Qinv_c = np.linalg.solve(Q, c)
    nu = np.linalg.solve(A @ QinvA_T, -(b + A @ Qinv_c))
    x = -np.linalg.solve(Q, c + A.T @ nu)
    return nu, qp_dual_function(Q, c, A, b, nu), x


def ridge_dual(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    K = X @ X.T
    a = np.linalg.solve(K + lam * np.eye(len(y)), y)
    return a, X.T @ a


def lasso_duality_gap(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    r = y - X @ w
    corr = float(np.max(np.abs(X.T @ r)))
    scale = 1.0 if corr <= lam else lam / corr
    theta = scale * r
    primal = 0.5 * float(r @ r) + lam * float(np.sum(np.abs(w)))
    dual = 0.5 * float(y @ y) - 0.5 * float(np.sum((y - theta) ** 2))
    return primal - dual


# Provided: FISTA for the lasso, to watch the gap shrink.
def fista_iterates(X: np.ndarray, y: np.ndarray, lam: float, iters: int) -> list[np.ndarray]:
    step = 1.0 / np.linalg.eigvalsh(X.T @ X).max()
    w = np.zeros(X.shape[1])
    z, t = w.copy(), 1.0
    out = [w.copy()]
    for _ in range(iters):
        u = z - step * X.T @ (X @ z - y)
        w_new = np.sign(u) * np.maximum(np.abs(u) - step * lam, 0.0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + (t - 1.0) / t_new * (w_new - w)
        w, t = w_new, t_new
        out.append(w.copy())
    return out


if __name__ == "__main__":
    f = lambda x: x**2
    g = lambda x: 1.0 - x
    print("L(1, 2) =", lagrangian(1.0, [2.0], [], f, [g], []), " L(3, 2) =", lagrangian(3.0, [2.0], [], f, [g], []))
    Q, c = np.eye(2), np.zeros(2)
    A, b = np.array([[1.0, 1.0]]), np.array([1.0])
    print("qp_dual:", qp_dual(Q, c, A, b))
    X, y = np.array([[1.0], [2.0]]), np.array([1.0, 2.0])
    print("ridge_dual:", ridge_dual(X, y, 1.0))
    for wv in (0.0, 0.4, 0.8):
        print(f"lasso gap at w = {wv}:", lasso_duality_gap(X, y, np.array([wv]), 1.0))
    print("kkt:", kkt_residuals(np.eye(2), np.zeros(2), np.array([[-1.0, -1.0]]), np.array([-1.0]),
                                np.array([0.5, 0.5]), np.array([0.5])))
    rng = np.random.default_rng(0)
    Xb = rng.normal(size=(100, 20))
    wt = np.zeros(20)
    wt[:4] = [3.0, -2.0, 1.5, 2.0]
    yb = Xb @ wt + 0.5 * rng.normal(size=100)
    path = fista_iterates(Xb, yb, 10.0, 40)
    print("FISTA gap at steps 0, 10, 20, 40:", [f"{lasso_duality_gap(Xb, yb, path[k], 10.0):.3g}" for k in (0, 10, 20, 40)])
