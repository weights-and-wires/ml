"""Kernels and the representer theorem.

Fill in the seven functions below. The first two build kernel matrices; `is_psd`
checks the property that makes a function a valid kernel at all.
`kernel_ridge_fit` and `kernel_ridge_predict` fit and use kernel ridge
regression. `random_fourier_features` builds an explicit,
finite-dimensional approximation to the RBF kernel's implicit feature map.
`representer_check` evaluates the regularised objective for a candidate
function given by its kernel-expansion coefficients over some point set,
which is what lets the exercise probe the representer theorem numerically.
"""

import numpy as np


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """Return the Gaussian (RBF) kernel matrix between two point sets.

    k(x, x') = exp(-gamma * ||x - x'||^2).

    Args:
        X1: Points, shape (n1, d).
        X2: Points, shape (n2, d).
        gamma: The kernel bandwidth parameter, gamma > 0.

    Returns:
        The Gram matrix, shape (n1, n2).
    """
    raise NotImplementedError("rbf_kernel")


def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, coef0: float = 1.0) -> np.ndarray:
    """Return the polynomial kernel matrix between two point sets.

    k(x, x') = (x . x' + coef0) ** degree.

    Args:
        X1: Points, shape (n1, d).
        X2: Points, shape (n2, d).
        degree: A positive integer degree.
        coef0: The additive constant inside the power.

    Returns:
        The Gram matrix, shape (n1, n2).
    """
    raise NotImplementedError("polynomial_kernel")


def is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Return whether K is a valid kernel matrix: square, symmetric, and PSD.

    Args:
        K: A candidate Gram matrix, shape (n, n).
        tol: Numerical tolerance. K is symmetric if it agrees with its
            transpose to within `tol`, and PSD if its smallest eigenvalue is
            at least `-tol`.

    Returns:
        True if K passes both tests.
    """
    raise NotImplementedError("is_psd")


def kernel_ridge_fit(K: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Fit kernel ridge regression: solve (K + lam I) alpha = y.

    Solve through a Cholesky factorisation of `K + lam * I`, which is always
    symmetric positive definite for `lam > 0` when `K` is PSD, rather than
    forming an explicit inverse.

    Args:
        K: Training Gram matrix, shape (n, n).
        y: Targets, shape (n,).
        lam: The regularisation strength, lam > 0.

    Returns:
        The coefficient vector alpha, shape (n,).
    """
    raise NotImplementedError("kernel_ridge_fit")


def kernel_ridge_predict(K_test: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Predict with a fitted kernel ridge model.

    Args:
        K_test: Kernel values between query points and training points,
            shape (n_test, n_train).
        alpha: Coefficients from `kernel_ridge_fit`, shape (n_train,).

    Returns:
        Predictions, shape (n_test,).
    """
    raise NotImplementedError("kernel_ridge_predict")


def random_fourier_features(X: np.ndarray, D: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Build a D-dimensional random Fourier feature map approximating the RBF kernel.

    Draw a (D, d) matrix of angular frequencies from N(0, 2 * gamma * I) and a
    length-D vector of phases from Uniform(0, 2*pi), then return

        Z(x) = sqrt(2 / D) * cos(omega @ x + phase) ,

    stacked over every row of X. On average over the random draws, `Z @ Z.T`
    equals the RBF Gram matrix, and the error shrinks as D grows.
    Draw omega first and the phases second, both from `rng`.

    Args:
        X: Points, shape (n, d).
        D: The number of random features.
        gamma: The RBF bandwidth, matching `rbf_kernel`'s `gamma`.
        rng: A NumPy random generator, used for both omega and the phases.

    Returns:
        The feature matrix, shape (n, D).
    """
    raise NotImplementedError("random_fourier_features")


def representer_check(K: np.ndarray, y: np.ndarray, lam: float, coeffs: np.ndarray) -> float:
    """Evaluate the regularised objective for a candidate kernel expansion.

    `K` is a Gram matrix over some point set, and `coeffs` gives the
    coefficients of a function f = sum_j coeffs[j] * k(., z_j) over that same
    point set. The first `len(y)` points are assumed to be the training
    points the targets `y` belong to; any further points (and further
    coefficients) let a caller probe functions that use kernel sections
    beyond the training set.

    Args:
        K: Gram matrix over N >= len(y) points, shape (N, N).
        y: Training targets, shape (n,), with n <= N.
        lam: The regularisation strength.
        coeffs: Coefficients of the candidate function, shape (N,).

    Returns:
        sum((y - f(x_1..n))**2) + lam * (squared size of f), as a float, where
        the squared size of f is coeffs @ K @ coeffs.
    """
    raise NotImplementedError("representer_check")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 30, 4
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + 0.1 * rng.normal(size=n)
    gamma, lam = 0.5, 0.2

    K = rbf_kernel(X, X, gamma)
    print(f"RBF Gram matrix is PSD: {is_psd(K)}")

    alpha = kernel_ridge_fit(K, y, lam)
    train_pred = kernel_ridge_predict(K, alpha)
    print(f"training RMSE: {np.sqrt(np.mean((train_pred - y) ** 2)):.4f}")

    for D in (50, 500, 5000):
        Z = random_fourier_features(X, D, gamma, np.random.default_rng(1))
        approx = Z @ Z.T
        error = np.abs(K - approx).mean()
        print(f"D={D:>5}: mean absolute error against the exact RBF Gram matrix = {error:.4f}")

    baseline = representer_check(K, y, lam, alpha)
    print(f"regularised objective at the representer-optimal alpha: {baseline:.4f}")
