"""Exercise: price the rules, bound the score, and measure the gap."""

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
    """Return f(x) + sum_i lam_i * g_i(x) + sum_j nu_j * h_j(x).

    Args:
        x: The point, any shape the functions accept.
        lam: Prices on the inequality rules g_i(x) <= 0, one per entry of gs.
        nu: Prices on the equality rules h_j(x) = 0, one per entry of hs.
        f: The score, a function of x that returns a number.
        gs: Inequality functions, each of x, each returning a number.
        hs: Equality functions, each of x, each returning a number.

    Returns:
        The priced score as a float.
    """
    raise NotImplementedError("write lagrangian")


def kkt_residuals(
    Q: np.ndarray, c: np.ndarray, G: np.ndarray, h: np.ndarray, x: np.ndarray, lam: np.ndarray
) -> dict:
    """Measure how far (x, lam) is from the KKT conditions of a small inequality problem.

    The problem is min 0.5 x^T Q x + c^T x subject to G x <= h.

    Args:
        Q: (d, d) positive definite matrix.
        c: (d,) vector.
        G: (m, d) rule matrix.
        h: (m,) rule limits.
        x: (d,) candidate point.
        lam: (m,) candidate prices.

    Returns:
        A dict with four floats, each 0 when the condition holds:
        "stationarity" is the largest |Q x + c + G^T lam| entry,
        "primal" is the largest positive entry of G x - h (0 if none),
        "dual" is the largest positive entry of -lam (0 if none), and
        "complementary" is the largest |lam_i * (G x - h)_i|.
    """
    raise NotImplementedError("write kkt_residuals")


def qp_dual_function(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, nu: np.ndarray) -> float:
    """Return g(nu), the lowest Lagrangian value, for min 0.5 x^T Q x + c^T x s.t. A x = b.

    Args:
        Q: (d, d) positive definite matrix.
        c: (d,) vector.
        A: (m, d) equality matrix.
        b: (m,) right-hand side.
        nu: (m,) prices on the equalities.

    Returns:
        g(nu) as a float.
    """
    raise NotImplementedError("write qp_dual_function")


def qp_dual(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Maximise the dual of min 0.5 x^T Q x + c^T x s.t. A x = b.

    Args:
        Q, c, A, b: As in qp_dual_function.

    Returns:
        (nu_star, d_star, x_star): the (m,) best prices, the best dual value, and the
        (d,) plan x that minimises the Lagrangian at nu_star.
    """
    raise NotImplementedError("write qp_dual")


def ridge_dual(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve ridge regression through its dual.

    Args:
        X: (n, d) design matrix.
        y: (n,) targets.
        lam: Ridge fee, positive.

    Returns:
        (a, w): the (n,) dual coefficients a = (X X^T + lam I)^-1 y, and the (d,)
        weights w = X^T a, which equal the primal ridge solution.
    """
    raise NotImplementedError("write ridge_dual")


def lasso_duality_gap(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    """Return P(w) - D(theta) for the lasso P(w) = 0.5 |Xw - y|^2 + lam |w|_1.

    Build theta from the residual y - X w, scaled down into |X^T theta|_inf <= lam.
    D(theta) = 0.5 |y|^2 - 0.5 |y - theta|^2.

    Args:
        X: (n, d) design matrix.
        y: (n,) targets.
        w: (d,) weights.
        lam: Lasso fee, positive.

    Returns:
        The duality gap as a float, never negative.
    """
    raise NotImplementedError("write lasso_duality_gap")


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
