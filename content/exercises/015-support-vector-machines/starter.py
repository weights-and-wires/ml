"""The widest road, its dual, and the two-at-a-time solver.

Fill in the ten functions below. Labels are in {-1, +1}. `alpha[i]` is how hard
house i pushes on the road, `K[i, j]` says how alike houses i and j are, and `C`
caps every push and prices every unit of slack.
"""

import numpy as np


def functional_margins(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return z_i = y_i (w . x_i + b) for every house.

    Args:
        w: Weights, shape (d,).
        b: Offset.
        X: Points, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).

    Returns:
        An array of shape (n,). A positive entry means the house is on its correct side.
    """
    raise NotImplementedError("functional_margins")


def hinge_loss_and_subgradient(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, C: float):
    """Return the soft-margin objective and its subgradient.

    The objective is 0.5 * ||w||^2 + C * sum_i max(0, 1 - z_i). A house with
    z_i exactly 1 is inactive.

    Args:
        w: Weights, shape (d,).
        b: Offset.
        X: Points, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).
        C: The price of one unit of slack.

    Returns:
        A tuple (loss, grad_w, grad_b): a float, an array of shape (d,), and a float.
    """
    raise NotImplementedError("hinge_loss_and_subgradient")


def fit_subgradient(X: np.ndarray, y: np.ndarray, C: float, lr: float = 0.3, steps: int = 3000):
    """Train a linear SVM by subgradient descent from w = 0, b = 0.

    At step t, subtract both slopes scaled by lr / (1 + lr * t). Compute both
    slopes before you update either parameter.

    Args:
        X: Points, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).
        C: The price of one unit of slack.
        lr: The first step size.
        steps: The number of steps.

    Returns:
        A tuple (w, b) with w of shape (d,).
    """
    raise NotImplementedError("fit_subgradient")


def dual_scores(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    """Return the score f_i = sum_k alpha_k y_k K[i, k] + b of every house.

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Offset.

    Returns:
        An array of shape (n,). With K = X @ X.T this equals X @ w + b, where
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


def margin_groups(alpha: np.ndarray, C: float, tol: float = 1e-8) -> np.ndarray:
    """Sort every house into one of three groups by the size of its push.

    Args:
        alpha: The pushes, shape (n,).
        C: The cap on every push.
        tol: Pushes within `tol` of 0 or of C count as being at that end.

    Returns:
        An integer array of shape (n,): 0 when the push is at 0, 1 when it is
        strictly between 0 and C, and 2 when it is at C.
    """
    raise NotImplementedError("margin_groups")


def kkt_violation(
    alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, tol: float = 1e-8
) -> float:
    """Return the largest violation of the three group rules.

    With g_i = y_i f_i - 1, a house with no push needs g >= 0, a free house
    needs g = 0, and a capped house needs g <= 0. A house's violation is its
    distance outside its band.

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Offset.
        C: The cap on every push.
        tol: Passed to `margin_groups`.

    Returns:
        One float, 0 when every house obeys its rule.
    """
    raise NotImplementedError("kkt_violation")


def clip_bounds(alpha: np.ndarray, y: np.ndarray, C: float, i: int, j: int):
    """Return (L, H), the ends of the allowed range of alpha_j when the pair (i, j) moves.

    The pair must keep y_i alpha_i + y_j alpha_j constant and stay in [0, C].

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        C: The cap on every push.
        i: Index of the first house.
        j: Index of the second house. The bounds are for alpha_j.

    Returns:
        Two floats, L and H.
    """
    raise NotImplementedError("clip_bounds")


def smo_step(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, i: int, j: int):
    """Move the pair (i, j) to the top of its parabola, clipped to [L, H].

    Returns the inputs unchanged when i == j, when eta = K_ii + K_jj - 2 K_ij
    is at most 1e-12, or when L >= H. Never changes `alpha` in place.

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Offset.
        C: The cap on every push.
        i: Index of the first house.
        j: Index of the second house.

    Returns:
        A tuple (new_alpha, new_b).
    """
    raise NotImplementedError("smo_step")


def fit_smo(K: np.ndarray, y: np.ndarray, C: float, max_iter: int = 200, tol: float = 1e-3):
    """Find the pushes by repeated passes of pair updates.

    In each pass, take every house i whose y_i * E_i breaks the checklist by
    more than `tol`, scan j from 0, and keep the first step that moves alpha_j
    by more than 1e-7. Stop after a pass in which `kkt_violation` is below
    `tol`, after a pass that changed nothing, or after `max_iter` passes.

    Args:
        K: Gram matrix, shape (n, n).
        y: Labels in {-1, +1}, shape (n,).
        C: The cap on every push.
        max_iter: The largest number of passes.
        tol: The stopping tolerance.

    Returns:
        A tuple (alpha, b) with alpha of shape (n,).
    """
    raise NotImplementedError("fit_smo")


def houses():
    """Twelve houses on a map, in units of 100 m. Circles are -1 and squares are +1."""
    X = np.array(
        [
            [3.0, 4.5], [2.7, 3.4], [2.5, 3.3], [3.1, 5.4], [2.5, 3.6], [3.5, 4.5],
            [8.1, 1.6], [8.0, 3.0], [6.7, 2.0], [6.1, 1.2], [6.2, 2.2], [6.7, 2.6],
        ]
    )
    y = np.array([-1.0] * 6 + [1.0] * 6)
    return X, y


if __name__ == "__main__":
    X, y = houses()
    C = 1.0

    # The road as weights, trained by the subgradient.
    w, b = fit_subgradient(X, y, C)
    z = functional_margins(w, b, X, y)
    loss, _, _ = hinge_loss_and_subgradient(w, b, X, y, C)
    print(f"w = {w}, b = {b:.3f}")
    print(f"smallest margin z = {z.min():.3f}, road half-width 1/||w|| = {1.0 / np.linalg.norm(w):.3f}")
    print(f"primal objective J = {loss:.4f}")

    # The same road as pushes, found two at a time.
    K = X @ X.T
    alpha, b2 = fit_smo(K, y, C, 500, 1e-6)
    w2 = (alpha * y) @ X
    print(f"pushes: {(alpha > 1e-6).sum()} support houses of {len(y)}")
    print(f"w from the pushes = {w2}, b = {b2:.3f}")
    print(f"dual objective D = {dual_objective(alpha, y, K):.4f}")
    print(f"KKT violation = {kkt_violation(alpha, y, K, b2, C):.2e}")
