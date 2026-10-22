"""ReLU, its slope, and merging two affine layers.

An affine layer maps a vector x to W @ x + b, with W of shape (out, in).
Fill in the three functions below, in order. The lesson walks through each one.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Return the elementwise ReLU, max(x, 0).

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape.
    """
    raise NotImplementedError("relu")


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Return the elementwise slope of ReLU.

    Args:
        x: An array of any shape.

    Returns:
        A float array of the same shape, 1.0 where x > 0 and 0.0 where x < 0.
        The value at x == 0 does not matter for the checks.
    """
    raise NotImplementedError("relu_grad")


def compose_affine(W1, b1, W2, b2):
    """Merge two affine layers into one.

    Args:
        W1: First layer weights, shape (mid, in).
        b1: First layer constants, shape (mid,).
        W2: Second layer weights, shape (out, mid).
        b2: Second layer constants, shape (out,).

    Returns:
        A pair (W, b) with W of shape (out, in) and b of shape (out,), such that
        W @ x + b equals W2 @ (W1 @ x + b1) + b2 for every x.
    """
    raise NotImplementedError("compose_affine")


if __name__ == "__main__":
    # XOR: lamp on when exactly one switch is up.
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    W_hidden = np.array([[1.0, -1.0], [-1.0, 1.0]])  # rows: x1 - x2 and x2 - x1
    hidden = relu(points @ W_hidden.T)
    print("hidden features:\n", hidden)
    print("h1 + h2 - 0.5:", hidden.sum(axis=1) - 0.5)

    W, b = compose_affine(
        np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0]),
        np.array([[3.0, 1.0]]), np.array([4.0]),
    )
    print("merged W:", W, "merged b:", b)
    print("relu slope on [2, -3, -2]:", relu_grad(np.array([2.0, -3.0, -2.0])))
