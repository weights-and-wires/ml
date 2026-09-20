"""Support vector machines: the hinge loss and a simplified SMO.

Fill in the four functions below. `hinge_loss_and_subgradient` is the
unconstrained equivalent of the soft-margin primal. `kkt_violation` measures
how far a candidate dual solution is from optimal. `smo_step` performs one
clipped two-variable update of the dual, and `fit_smo` drives a sequence of
those steps to convergence.

Throughout, `K` is a precomputed Gram matrix: `K[i, j]` is the kernel value
between training points `i` and `j`. For a linear kernel, `K = X @ X.T`, but
none of these functions need to know that; they only ever read `K`.
"""

import numpy as np


def hinge_loss_and_subgradient(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, C: float):
    """Evaluate the soft-margin objective and a subgradient at (w, b).

    The objective is

        0.5 * ||w||^2 + C * sum_i max(0, 1 - y_i (w . x_i + b)) .

    Args:
        w: Weight vector, shape (d,).
        b: Bias term.
        X: Training features, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).
        C: The slack penalty.

    Returns:
        A tuple `(loss, grad_w, grad_b)`. `grad_w` has shape (d,); `grad_b`
        is a float. At a point where some margin is exactly 1, return the
        subgradient that treats that point as if it were not active (its
        contribution to the sum is zero), which is one valid member of the
        subdifferential.
    """
    raise NotImplementedError("hinge_loss_and_subgradient")


def kkt_violation(
    alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, tol: float = 1e-8
) -> float:
    """Return the largest KKT violation among the dual variables.

    Let `f_i = sum_k alpha_k y_k K[i, k] + b` and `g_i = y_i f_i - 1`. The
    KKT conditions classify point `i` by its multiplier:

        alpha_i ~ 0        requires g_i >= 0  (outside or on the margin)
        0 < alpha_i < C    requires g_i == 0  (exactly on the margin)
        alpha_i ~ C        requires g_i <= 0  (inside the margin, or violated)

    where "~" means within `tol`. The violation for point `i` is how far it
    falls short of its required condition: `max(0, -g_i)` in the first case,
    `max(0, g_i)` in the third, and `abs(g_i)` in the second.

    Args:
        alpha: Dual variables, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Bias term.
        C: The slack penalty, an upper bound on alpha.
        tol: Tolerance for deciding whether alpha_i is at a boundary.

    Returns:
        The largest violation over all i, as a float. Zero means every KKT
        condition holds exactly.
    """
    raise NotImplementedError("kkt_violation")


def smo_step(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, i: int, j: int):
    """Perform one clipped SMO update on the pair (alpha_i, alpha_j).

    Holding every other alpha_k fixed, the equality constraint
    `sum_k alpha_k y_k = 0` pins `alpha_i` to an affine function of
    `alpha_j`. Substituting that relationship turns the dual objective,
    restricted to this pair, into a one-dimensional quadratic in `alpha_j`.
    Maximise it in closed form, then clip the result to the box the equality
    constraint and `0 <= alpha <= C` cut out of the line, and recover
    `alpha_i` from the clipped `alpha_j`. Finally update `b` from whichever
    of the two points ends up strictly inside `(0, C)`, averaging the two
    candidate values if neither does.

    Args:
        alpha: Current dual variables, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Current bias term.
        C: The slack penalty.
        i: Index of the first variable to update.
        j: Index of the second variable to update.

    Returns:
        A tuple `(new_alpha, new_b)`. `new_alpha` is a full-length array that
        agrees with `alpha` everywhere except possibly at `i` and `j`. When
        `i == j`, or when the two points coincide in feature space (so the
        update direction is degenerate), return `alpha` and `b` unchanged.
    """
    raise NotImplementedError("smo_step")


def fit_smo(K: np.ndarray, y: np.ndarray, C: float, max_iter: int = 200, tol: float = 1e-3):
    """Fit the soft-margin dual by repeated two-variable SMO updates.

    Start from alpha = 0, b = 0. Repeatedly scan the points in index order;
    for each point whose KKT condition is violated by more than `tol`, pick
    a second index by scanning `j` from 0 and take the first `j` for which
    `smo_step` actually changes `alpha_j`. Stop when a full scan makes no
    change, when `kkt_violation` drops below `tol`, or after `max_iter`
    outer scans, whichever comes first.

    Args:
        K: Gram matrix, shape (n, n).
        y: Labels in {-1, +1}, shape (n,).
        C: The slack penalty.
        max_iter: The maximum number of outer scans over the data.
        tol: The KKT tolerance that decides convergence.

    Returns:
        A tuple `(alpha, b)` with `alpha` of shape (n,).
    """
    raise NotImplementedError("fit_smo")


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
