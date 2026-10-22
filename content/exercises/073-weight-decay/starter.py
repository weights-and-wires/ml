"""Weight decay: a pull toward zero on every gradient step.

Fill in the three functions below.
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
