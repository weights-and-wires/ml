"""Starter for the ICA and t-SNE exercise. Fill in every function that raises NotImplementedError."""

from __future__ import annotations

import numpy as np


def whiten(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """PCA-whiten the rows of X. Given in full, as in the principal components chapter.

    Args:
        X: Data of shape (n, d).
        eps: Added to each eigenvalue before the square root.

    Returns:
        Centred data of shape (n, d) whose covariance is the identity.
    """
    Xc = X - X.mean(axis=0)
    lam, U = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    return Xc @ U / np.sqrt(lam + eps)


def contrast_terms(u: np.ndarray, g: str) -> tuple[np.ndarray, np.ndarray]:
    """Return g(u) and g'(u) for a contrast.

    Args:
        u: Projections, any shape.
        g: "logcosh" (g = tanh, g' = 1 - tanh^2) or "kurtosis" (g = u^3, g' = 3 u^2).

    Returns:
        A pair (g(u), g'(u)), each with the shape of u.
    """
    raise NotImplementedError("write contrast_terms")


def fastica_one_unit(Xw: np.ndarray, w0: np.ndarray, g: str = "logcosh", steps: int = 100) -> np.ndarray:
    """Run the one-unit FastICA fixed-point update on whitened data.

    Args:
        Xw: Whitened data, shape (n, d).
        w0: Starting direction, shape (d,).
        g: Contrast name for contrast_terms.
        steps: Number of updates.

    Returns:
        A unit vector of shape (d,).
    """
    raise NotImplementedError("write fastica_one_unit")


def fastica(Xw: np.ndarray, k: int, rng: np.random.Generator, g: str = "logcosh", steps: int = 200) -> np.ndarray:
    """Find k unmixing directions one at a time, with deflation.

    Args:
        Xw: Whitened data, shape (n, d).
        k: Number of directions.
        rng: Random generator for the starting directions.
        g: Contrast name.
        steps: Updates per direction.

    Returns:
        W of shape (k, d) with orthonormal rows. The sources are Xw @ W.T.
    """
    raise NotImplementedError("write fastica")


def conditional_p(D: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Turn squared distances into conditional probabilities p(j | i).

    Args:
        D: Squared distances, shape (n, n).
        betas: Precision 1 / (2 sigma_i^2) per row, shape (n,).

    Returns:
        P of shape (n, n), zero diagonal, each row summing to 1.
    """
    raise NotImplementedError("write conditional_p")


def perplexity_bandwidths(D: np.ndarray, perplexity: float, tol: float = 1e-6, iters: int = 200) -> np.ndarray:
    """Find each row's beta so that its entropy in bits equals log2(perplexity).

    Args:
        D: Squared distances, shape (n, n).
        perplexity: Target perplexity, between 1 and n - 1.
        tol: Stop when the entropy is this close to the target.
        iters: Maximum bisection steps per row.

    Returns:
        betas of shape (n,).
    """
    raise NotImplementedError("write perplexity_bandwidths")


def joint_p(D: np.ndarray, perplexity: float) -> np.ndarray:
    """Symmetrise the conditionals into one joint distribution.

    Args:
        D: Squared distances, shape (n, n).
        perplexity: Target perplexity.

    Returns:
        P of shape (n, n), symmetric, zero diagonal, summing to 1.
    """
    raise NotImplementedError("write joint_p")


def student_q(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Student-t (one degree of freedom) similarities of the map points.

    Args:
        Y: Map points, shape (n, 2).

    Returns:
        (Q, inv): inv[i, j] = 1 / (1 + |y_i - y_j|^2) with a zero diagonal, and
        Q = inv / inv.sum(). Both have shape (n, n).
    """
    raise NotImplementedError("write student_q")


def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """KL(P || Q) over the off-diagonal entries, skipping entries where P is 0."""
    raise NotImplementedError("write kl_divergence")


def tsne_grad(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Gradient of KL(P || Q) with respect to the map points.

    Args:
        P: Joint probabilities, shape (n, n).
        Y: Map points, shape (n, 2).

    Returns:
        Gradient of shape (n, 2).
    """
    raise NotImplementedError("write tsne_grad")


def tsne_embed(
    P: np.ndarray,
    Y0: np.ndarray,
    steps: int = 300,
    lr: float = 100.0,
    momentum: float = 0.8,
    exaggeration: float = 12.0,
    exag_steps: int = 100,
) -> np.ndarray:
    """Gradient descent with momentum on the KL, with early exaggeration.

    Args:
        P: Joint probabilities, shape (n, n).
        Y0: Starting map, shape (n, 2).
        steps: Total updates.
        lr: Step size.
        momentum: Weight on the previous update.
        exaggeration: Factor that multiplies P for the first exag_steps updates.
        exag_steps: How many updates use the exaggerated P.

    Returns:
        The final map, shape (n, 2).
    """
    raise NotImplementedError("write tsne_embed")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    S = np.column_stack([rng.uniform(-1, 1, 500), rng.laplace(size=500)])
    X = S @ np.array([[1.0, 0.5], [0.4, 1.0]]).T
    Xw = whiten(X)
    W = fastica(Xw, 2, rng)
    print("unmixing rows:\n", W)
    print("corr with sources:\n", np.round(np.corrcoef((Xw @ W.T).T, S.T)[:2, 2:], 3))
