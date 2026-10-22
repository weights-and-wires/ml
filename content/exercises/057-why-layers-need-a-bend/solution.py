"""Reference solution: ReLU, its slope, and merging two affine layers."""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def compose_affine(W1, b1, W2, b2):
    # W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
    return W2 @ W1, W2 @ b1 + b2


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
