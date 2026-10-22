"""Early stopping as a ridge penalty.

Fill in the three functions below.
"""

import numpy as np


def gd_quadratic_1d(eigenvalue: float, step: float, w_star: float, iterations: int) -> float:
    """Run gradient descent from zero on one direction of a quadratic loss.

    The loss is `0.5 * eigenvalue * (w - w_star) ** 2`. Use a loop, so the
    function also works when a step overshoots.

    Args:
        eigenvalue: The curvature along this direction, positive.
        step: The step size.
        w_star: The weight that minimises the loss.
        iterations: The number of gradient steps, a non-negative integer.

    Returns:
        The weight after `iterations` steps, as a float. It is 0.0 when
        `iterations` is 0.
    """
    raise NotImplementedError("gd_quadratic_1d")


def early_stopping_equivalent_ridge(eigenvalue: float, step: float, iterations: int) -> float:
    """Return the ridge penalty that `iterations` gradient steps match.

    On one direction with curvature `eigenvalue`, ridge with penalty `lam`
    keeps the share `eigenvalue / (eigenvalue + lam)` of `w_star`. Gradient
    descent from zero reaches the share `1 - (1 - step * eigenvalue) ** iterations`.
    Return the `lam` that makes the two shares equal.

    Args:
        eigenvalue: The curvature along this direction, positive.
        step: The step size.
        iterations: The number of gradient steps, a non-negative integer.

    Returns:
        The penalty as a float. It is `float("inf")` when `iterations` is 0.

    Raises:
        ValueError: If `step * eigenvalue` is not in the range (0, 1]. Outside
            that range no positive penalty matches gradient descent.
    """
    raise NotImplementedError("early_stopping_equivalent_ridge")


def best_stopping_step(val_losses, patience: int) -> int:
    """Return the step early stopping would keep, given a validation curve.

    Scan `val_losses` in order. A loss that is strictly lower than the best so
    far is a new best and resets the count of checks since the last best. Any
    other loss adds one to that count. Stop scanning when the count reaches
    `patience`.

    Args:
        val_losses: A sequence of validation losses, one per check, not empty.
        patience: The number of checks in a row without a new best that ends
            the scan, at least 1.

    Returns:
        The index of the best loss seen before the scan stopped. Ties keep the
        earlier index.
    """
    raise NotImplementedError("best_stopping_step")


if __name__ == "__main__":
    c, a, w_star = 2.0, 0.1, 1.0
    print("k   gradient descent   ridge solution   penalty")
    for k in (0, 1, 10, 50):
        w = gd_quadratic_1d(c, a, w_star, k)
        lam = early_stopping_equivalent_ridge(c, a, k)
        ridge = 0.0 if np.isinf(lam) else c / (c + lam) * w_star
        print(f"{k:<3} {w:>16.5f} {ridge:>16.5f} {lam:>10.5f}")

    losses = [5.0, 4.0, 4.5, 3.0, 3.5, 3.6, 3.7]
    for patience in (1, 2):
        print(f"patience {patience}: stop at step {best_stopping_step(losses, patience)}")
