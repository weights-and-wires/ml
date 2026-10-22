"""Sort the houses and run the KKT checklist.

Fill in the three functions below. Labels are in {-1, +1}. `alpha[i]` is how hard
house i pushes on the road, `C` caps every push, and `K` is the Gram matrix.
"""

import numpy as np


# Written in the chapter "Pushes from each point: the dual and kernels".
def dual_scores(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    return (alpha * y) @ K + b


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


def slack_amounts(alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
    """Return how far each house falls short of the road's edge, max(0, 1 - y_i f_i).

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Offset.

    Returns:
        An array of shape (n,) with no negative entries.
    """
    raise NotImplementedError("slack_amounts")


def kkt_violation(
    alpha: np.ndarray, y: np.ndarray, K: np.ndarray, b: float, C: float, tol: float = 1e-8
) -> float:
    """Return the largest amount by which any house breaks the rule for its group.

    With g_i = y_i f_i - 1 and f_i the score from `dual_scores`:

        group 0 (no push)        needs g_i >= 0, so it violates by max(0, -g_i)
        group 1 (0 < alpha < C)  needs g_i == 0, so it violates by abs(g_i)
        group 2 (capped at C)    needs g_i <= 0, so it violates by max(0, g_i)

    Args:
        alpha: The pushes, shape (n,).
        y: Labels in {-1, +1}, shape (n,).
        K: Gram matrix, shape (n, n).
        b: Offset.
        C: The cap on every push.
        tol: Passed to `margin_groups`.

    Returns:
        The largest violation over all houses, as a float. Zero means the pushes
        meet the checklist.
    """
    raise NotImplementedError("kkt_violation")


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
