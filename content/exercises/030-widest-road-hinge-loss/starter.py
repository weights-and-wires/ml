"""The widest road, trained with the hinge loss.

Fill in the three functions below. Labels are in {-1, +1}. `X` has one house per
row, `w` is the road's normal vector, and `b` is its offset.
"""

import numpy as np


def functional_margins(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the functional margin z_i = y_i (w . x_i + b) of every point.

    Args:
        w: Weight vector, shape (d,).
        b: Offset.
        X: Points, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).

    Returns:
        An array of shape (n,). Positive means the point is on its correct side,
        and 1 means it rests exactly on the edge of the road.
    """
    raise NotImplementedError("functional_margins")


def hinge_loss_and_subgradient(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, C: float):
    """Evaluate the soft-margin objective and one subgradient at (w, b).

    The objective is

        J(w, b) = 0.5 * ||w||^2 + C * sum_i max(0, 1 - z_i),   z_i = y_i (w . x_i + b).

    The sum runs over points, it is not a mean. A point with z_i < 1 is active
    and pushes on the road. A point with z_i >= 1, including z_i = 1 exactly,
    is inactive and contributes nothing to the slope.

    Args:
        w: Weight vector, shape (d,).
        b: Offset.
        X: Points, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).
        C: The price per unit of shortfall.

    Returns:
        A tuple `(loss, grad_w, grad_b)`: a float, an array of shape (d,), and a float.
    """
    raise NotImplementedError("hinge_loss_and_subgradient")


def fit_subgradient(X: np.ndarray, y: np.ndarray, C: float, lr: float = 0.3, steps: int = 3000):
    """Train a linear SVM by subgradient descent on J.

    Start from w = 0 and b = 0. At step t (counting from 0), move both w and b
    against their subgradients by the step size lr / (1 + lr * t), so the steps
    shrink as training goes on.

    Args:
        X: Points, shape (n, d).
        y: Labels in {-1, +1}, shape (n,).
        C: The price per unit of shortfall.
        lr: The first step size.
        steps: The number of steps.

    Returns:
        A tuple `(w, b)` with `w` of shape (d,) and `b` a float.
    """
    raise NotImplementedError("fit_subgradient")


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
    w, b = fit_subgradient(X, y, C)
    z = functional_margins(w, b, X, y)
    loss, _, _ = hinge_loss_and_subgradient(w, b, X, y, C)
    print(f"w = {w}, b = {b:.3f}")
    print(f"smallest margin z = {z.min():.3f} (a value near 1 means a point rests on the road edge)")
    print(f"road half-width 1/||w|| = {1.0 / np.linalg.norm(w):.3f}")
    print(f"objective J = {loss:.4f}")
