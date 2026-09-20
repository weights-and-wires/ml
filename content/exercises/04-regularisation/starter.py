"""Ridge, lasso, and the proximal view.

Fill in the six functions below. The first three build the ridge path: a
closed-form solve, the same path read off cheaply from one SVD, and the
effective degrees of freedom that path costs. The next three build the lasso
by proximal gradient descent: the soft-thresholding operator, ISTA, and its
accelerated cousin FISTA.
"""

from __future__ import annotations

import numpy as np


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve ridge regression by one linear solve.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Ridge penalty strength, lam >= 0.

    Returns:
        w = (X^T X + lam I)^{-1} X^T y, shape (d,).
    """
    raise NotImplementedError("ridge_closed_form")


def ridge_svd_path(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Compute the ridge solution at every lambda from one SVD of X.

    X = U @ diag(s) @ Vt is the thin SVD of the design matrix. In this basis
    the ridge solution at penalty lambda is
    V @ diag(s / (s**2 + lambda)) @ U^T @ y; computing U^T y once and reusing
    it for every lambda is what makes the whole path cheap.

    Args:
        U: Left singular vectors, shape (n, r).
        s: Singular values, shape (r,).
        Vt: Right singular vectors transposed, shape (r, d).
        y: Targets, shape (n,).
        lambdas: Ridge penalties to evaluate, shape (m,).

    Returns:
        Coefficients at every penalty, shape (m, d).
    """
    raise NotImplementedError("ridge_svd_path")


def effective_degrees_of_freedom(s: np.ndarray, lam) -> np.ndarray | float:
    """Return sum_i s_i^2 / (s_i^2 + lam), the effective degrees of freedom.

    Args:
        s: Singular values of the design matrix, shape (r,).
        lam: A single ridge penalty (float) or an array of them, shape (m,).

    Returns:
        A float if lam is a scalar, or an array of shape (m,) if lam is an
        array, giving one degrees-of-freedom value per penalty.
    """
    raise NotImplementedError("effective_degrees_of_freedom")


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Apply the soft-thresholding operator elementwise.

    Args:
        x: Input array, any shape.
        thresh: Non-negative threshold.

    Returns:
        sign(x) * max(|x| - thresh, 0), elementwise, same shape as x.
    """
    raise NotImplementedError("soft_threshold")


def ista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the lasso objective 0.5 ||Xw - y||^2 + lam ||w||_1 with ISTA.

    Each step takes a gradient step on the smooth term and applies
    soft-thresholding for the l1 term: this is the proximal-gradient update.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Lasso penalty strength, lam >= 0.
        step: Fixed step size; use step <= 1 / L with L the largest
            eigenvalue of X^T X for the iteration to be guaranteed to
            decrease the objective.
        iters: Number of ISTA steps to take.
        w0: Optional starting point, shape (d,). Defaults to zeros.

    Returns:
        A tuple (w, history), where w is the final iterate, shape (d,), and
        history[k] is the lasso objective after k steps, shape (iters + 1,),
        with history[0] the objective at the starting point.
    """
    raise NotImplementedError("ista")


def fista(
    X: np.ndarray, y: np.ndarray, lam: float, step: float, iters: int, w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the lasso objective with FISTA, Nesterov's accelerated ISTA.

    Args:
        X: Design matrix, shape (n, d).
        y: Targets, shape (n,).
        lam: Lasso penalty strength, lam >= 0.
        step: Fixed step size, as in `ista`.
        iters: Number of FISTA steps to take.
        w0: Optional starting point, shape (d,). Defaults to zeros.

    Returns:
        A tuple (w, history), with the same meaning as in `ista`.
    """
    raise NotImplementedError("fista")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 100, 40
    latent = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = latent @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[:5] = rng.normal(size=5) * 2.0
    y = X @ w_true + 0.1 * rng.normal(size=n)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.array([0.1, 1.0, 10.0, 100.0])
    path = ridge_svd_path(U, s, Vt, y, lambdas)
    dof = effective_degrees_of_freedom(s, lambdas)
    print("ridge path, effective degrees of freedom:")
    for lam, w, df in zip(lambdas, path, dof):
        print(f"  lambda={lam:>7.2f}  dof={df:5.2f}  ||w||={np.linalg.norm(w):.3f}")

    L = np.linalg.eigvalsh(X.T @ X).max()
    lam_lasso = 3.0
    _, ista_history = ista(X, y, lam_lasso, 1.0 / L, 150)
    _, fista_history = fista(X, y, lam_lasso, 1.0 / L, 150)
    print(f"\nafter 150 steps: ISTA objective {ista_history[-1]:.4f}, FISTA {fista_history[-1]:.4f}")
