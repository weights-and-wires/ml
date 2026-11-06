"""The SMO solver, two pushes at a time.

Fill in the three functions below. Labels are in {-1, +1}. `alpha[i]` is how hard
house i pushes on the road, `C` caps every push, and `K` is the Gram matrix. The
four helpers at the top are the functions you wrote in the earlier chapters.
"""

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
    """Return (L, H), the ends of the allowed range for alpha_j when the pair moves.

    The pair must keep y_i alpha_i + y_j alpha_j constant, and both pushes must
    stay in [0, C]. When y_i != y_j the line is alpha_i - alpha_j = constant, and
    when y_i == y_j it is alpha_i + alpha_j = constant. Intersect that line with
    the square [0, C] x [0, C] and read off the ends of alpha_j.

    Args:
        alpha: The pushes before the step, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        C: The cap on every push.
        i: Index of the first house.
        j: Index of the second house.

    Returns:
        A tuple of two floats `(L, H)` with L <= H.
    """
    raise NotImplementedError("clip_bounds")


def smo_step(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, i: int, j: int):
    """Raise the dual as far as one pair of pushes allows, and update the bias.

    With E_i = f_i - y_i and eta = K[i, i] + K[j, j] - 2 K[i, j], the top of the
    parabola in alpha_j is alpha_j + y_j (E_i - E_j) / eta. Clip it to [L, H],
    then set alpha_i = alpha_i + y_i y_j (alpha_j_old - alpha_j_new) so the pair
    stays balanced. For the bias, b1 puts house i exactly on its edge and b2 does
    the same for house j:

        b1 = b - E_i - y_i d_i K[i, i] - y_j d_j K[i, j]
        b2 = b - E_j - y_i d_i K[i, j] - y_j d_j K[j, j]

    where d_i and d_j are the changes in alpha_i and alpha_j. Use b1 when the new
    alpha_i is strictly between 0 and C, else b2 when the new alpha_j is, else
    the average of b1 and b2. Read both errors before you write either push.

    Args:
        alpha: The pushes before the step, shape (n,). Do not modify it.
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: The bias before the step.
        C: The cap on every push.
        i: Index of the first house.
        j: Index of the second house.

    Returns:
        A tuple `(new_alpha, new_b)`. When i == j, when eta <= 1e-12, or when
        L >= H, return a copy of `alpha` and `b` unchanged.
    """
    raise NotImplementedError("smo_step")


def fit_smo(K: np.ndarray, y: np.ndarray, C: float, max_iter: int = 200, tol: float = 1e-3):
    """Solve the dual by repeated calls to `smo_step`.

    Start from alpha = 0 and b = 0. Make passes over the houses in index order.
    House i needs attention when y_i E_i < -tol and alpha_i < C, or when
    y_i E_i > tol and alpha_i > 0, where E_i = f_i - y_i. For such a house, scan
    j from 0 (skipping i) and keep the first `smo_step` that moves alpha_j by more
    than 1e-7. Stop after a pass in which `kkt_violation` (with its default
    tolerance) is below `tol`, after a pass that changed nothing, or after
    `max_iter` passes.

    Args:
        K: Gram matrix, shape (n, n).
        y: Labels in {-1, +1}, shape (n,).
        C: The cap on every push.
        max_iter: The maximum number of passes.
        tol: How small the largest violation must be to stop.

    Returns:
        A tuple `(alpha, b)` with `alpha` of shape (n,).
    """
    raise NotImplementedError("fit_smo")


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
