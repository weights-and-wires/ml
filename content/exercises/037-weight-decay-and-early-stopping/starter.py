"""Weight decay and early stopping: two ways to keep weights small.

Fill in the six functions below.
"""

import numpy as np


def l2_penalty_grad(w: np.ndarray, lam: float) -> np.ndarray:
    """Return the slope of the ridge fee (lam / 2) * sum(w ** 2) with respect to `w`.

    Args:
        w: Weights, any shape.
        lam: Pull strength, a non-negative float.

    Returns:
        An array with the same shape as `w`, holding the slope of the fee
        with respect to each weight.
    """
    raise NotImplementedError("l2_penalty_grad")


def weight_decay_step(w: np.ndarray, grad: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    """Take one gradient step on the loss plus the ridge fee.

    Build the step from `l2_penalty_grad`, so the pull and the data gradient
    move the weights together.

    Args:
        w: Current weights.
        grad: Gradient of the data loss alone at `w`, same shape as `w`.
        lr: Learning rate.
        weight_decay: Pull strength, the `lam` of the ridge fee.

    Returns:
        The new weights, same shape as `w`.
    """
    raise NotImplementedError("weight_decay_step")


def step_angle(w: np.ndarray, grad: np.ndarray, lr: float) -> float:
    """Return the angle one gradient step turns a weight vector through.

    The weights feed a normalisation layer, so `grad` is at a right angle to
    `w`. The step is sideways, and the angle is the one whose tangent is the
    step length divided by the length of `w`.

    Args:
        w: A weight vector of shape `(n,)`, not all zero.
        grad: A gradient of shape `(n,)` at a right angle to `w`.
        lr: Learning rate.

    Returns:
        The angle in radians, as a float.
    """
    raise NotImplementedError("step_angle")


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
    w = np.array([1.0])
    print("worked step:", weight_decay_step(w, np.array([0.1]), lr=0.1, weight_decay=0.01))

    w = np.array([1.0])
    for _ in range(30):
        w = weight_decay_step(w, np.zeros(1), lr=0.5, weight_decay=0.2)
    print(f"no data gradient, 30 steps: {w[0]:.4f} (0.9**30 = {0.9 ** 30:.4f})")

    w = np.array([1.0])
    for _ in range(200):
        w = weight_decay_step(w, w - 0.6, lr=0.5, weight_decay=0.2)
    print(f"data pulls to 0.6, decay 0.2: settles at {w[0]:.4f}")

    for length in (4.0, 2.0):
        angle = step_angle(np.array([length, 0.0]), np.array([0.0, 1.0]), lr=1.0)
        print(f"|w| = {length}: one sideways step turns it {np.degrees(angle):.1f} degrees")

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
