"""Kernel ridge regression.

Fill in the three functions below. `kernel_ridge_fit` finds the weights alpha
that solve (K + lam I) alpha = y. `kernel_ridge_predict` turns a matrix of
query-to-training similarities into predictions. `implied_weights` recovers
the ordinary ridge weight vector from alpha when the kernel is linear.
"""

import numpy as np


# Implemented in the exercise for "Kernels: a similarity score that hides a bigger
# feature space". Provided here so you can build kernel matrices.
def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    sq = (X1**2).sum(axis=1)[:, None] + (X2**2).sum(axis=1)[None, :] - 2.0 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def kernel_ridge_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Return the kernel ridge weights alpha that solve (K + lam I) alpha = y.

    The fitted model predicts sum_i alpha_i k(x, x_i). Solve with a Cholesky
    factor of K + lam I rather than an explicit inverse.

    Args:
        K: Gram matrix of the training points, shape (n, n), positive semidefinite.
        y: Training targets, shape (n,).
        lam: Ridge strength, lam > 0.

    Returns:
        The weights alpha, shape (n,).
    """
    raise NotImplementedError("kernel_ridge_fit")


def kernel_ridge_predict(K_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return predictions sum_i alpha_i k(x, x_i) for each query row.

    Args:
        K_test: Similarities from each query to each training point,
            shape (n_test, n_train).
        alpha: Weights from `kernel_ridge_fit`, shape (n_train,).

    Returns:
        Predictions, shape (n_test,).
    """
    raise NotImplementedError("kernel_ridge_predict")


def implied_weights(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Return the feature-space weights w = X^T alpha for a linear kernel.

    When K = X X^T, kernel ridge regression and ordinary ridge regression give
    the same predictions, and this w is the ordinary ridge weight vector.

    Args:
        X: Training inputs, shape (n, d).
        alpha: Weights from `kernel_ridge_fit`, shape (n,).

    Returns:
        The weight vector, shape (d,).
    """
    raise NotImplementedError("implied_weights")


if __name__ == "__main__":
    x = np.array([0.51, 2.37, 3.21, 3.82, 5.93, 6.13])[:, None]
    y = np.array([0.36, 0.73, -0.39, -0.58, -0.30, 0.17])
    gamma, lam = 1.0, 0.1

    K = rbf_kernel(x, x, gamma)
    alpha = kernel_ridge_fit(K, y, lam)
    train_pred = kernel_ridge_predict(K, alpha)
    print("alpha:", np.round(alpha, 3))
    print(f"training RMSE: {np.sqrt(np.mean((train_pred - y) ** 2)):.4f}")
    print("miss equals lam * alpha:", np.allclose(y - train_pred, lam * alpha))

    x_query = np.array([[1.5], [3.0], [5.0]])
    print("predictions at 1.5, 3.0, 5.0:", np.round(kernel_ridge_predict(rbf_kernel(x_query, x, gamma), alpha), 3))

    X = np.array([[1.0], [2.0]])
    a = kernel_ridge_fit(X @ X.T, np.array([1.0, 2.0]), 1.0)
    print("linear kernel: alpha =", np.round(a, 4), " w =", implied_weights(X, a), " (ordinary ridge gives 5/6)")
