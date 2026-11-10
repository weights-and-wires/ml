"""Reference solution: the widest road, its dual, and the two-at-a-time solver."""

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
    return y * (X @ w + b)


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
    z = functional_margins(w, b, X, y)
    loss = 0.5 * float(w @ w) + C * float(np.sum(np.maximum(0.0, 1.0 - z)))
    active = z < 1.0
    grad_w = w - C * (y[active] @ X[active])
    grad_b = -C * float(np.sum(y[active]))
    return loss, grad_w, grad_b


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
    w = np.zeros(X.shape[1])
    b = 0.0
    for t in range(steps):
        _, grad_w, grad_b = hinge_loss_and_subgradient(w, b, X, y, C)
        step = lr / (1.0 + lr * t)
        w = w - step * grad_w
        b = b - step * grad_b
    return w, b


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
    return (alpha * y) @ K + b


def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """Return D(alpha) = sum_i alpha_i - 0.5 * sum_ij alpha_i alpha_j y_i y_j K[i, j].

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).

    Returns:
        One float.
    """
    v = alpha * y
    return float(alpha.sum() - 0.5 * v @ K @ v)


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
    groups = np.ones(len(alpha), dtype=int)
    groups[alpha <= tol] = 0
    groups[alpha >= C - tol] = 2
    return groups


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
    g = y * dual_scores(alpha, y, K, b) - 1.0
    groups = margin_groups(alpha, C, tol)
    violation = np.where(
        groups == 0,
        np.maximum(0.0, -g),
        np.where(groups == 1, np.abs(g), np.maximum(0.0, g)),
    )
    return float(violation.max()) if len(violation) else 0.0


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
    if y[i] != y[j]:
        low = max(0.0, alpha[j] - alpha[i])
        high = min(C, C + alpha[j] - alpha[i])
    else:
        low = max(0.0, alpha[i] + alpha[j] - C)
        high = min(C, alpha[i] + alpha[j])
    return float(low), float(high)


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
    # Rounding can leave 7e-18 where the push is really 0, so snap it onto the wall.
    ai_new = 0.0 if ai_new < 1e-12 else (C if ai_new > C - 1e-12 else ai_new)
    aj_new = 0.0 if aj_new < 1e-12 else (C if aj_new > C - 1e-12 else aj_new)
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
