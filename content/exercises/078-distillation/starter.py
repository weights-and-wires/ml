"""Exercise: distillation. Soften a softmax, build the distillation loss and its gradient, and train a student."""

from __future__ import annotations

import numpy as np


def soft_targets(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Softmax of logits / T along the last axis.

    Args:
        logits: Array of shape (K,) or (n, K).
        T: Temperature, a positive number.

    Returns:
        Probabilities with the same shape as logits. Each last-axis slice sums to 1.
    """
    raise NotImplementedError("write soft_targets")


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) = sum_k p_k log(p_k / q_k) along the last axis.

    Args:
        p: Probabilities of shape (K,) or (n, K). A zero entry contributes 0.
        q: Probabilities of the same shape, positive wherever p is positive.

    Returns:
        Array of shape () or (n,), one divergence per row.
    """
    raise NotImplementedError("write kl_divergence")


def distill_loss(z_s: np.ndarray, z_t: np.ndarray, y: np.ndarray, T: float, alpha: float) -> float:
    """Mean over rows of alpha * T^2 * KL(p_t^T || p_s^T) + (1 - alpha) * CE(p_s, y).

    Args:
        z_s: Student logits, shape (n, K).
        z_t: Teacher logits, shape (n, K).
        y: Integer hard labels, shape (n,).
        T: Temperature for the soft term. The hard term uses temperature 1.
        alpha: Weight of the soft term, between 0 and 1.

    Returns:
        A float.
    """
    raise NotImplementedError("write distill_loss")


def distill_grad(z_s: np.ndarray, z_t: np.ndarray, y: np.ndarray, T: float, alpha: float) -> np.ndarray:
    """Gradient of distill_loss with respect to z_s, shape (n, K)."""
    raise NotImplementedError("write distill_grad")


def logit_match_grad(z_s: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    """Limit of the gradient of T^2 * KL(p_t^T || p_s^T) as T grows, for one row or many.

    Args:
        z_s: Student logits, shape (K,) or (n, K).
        z_t: Teacher logits, the same shape.

    Returns:
        Array of the same shape, written in terms of the centred logits of each row.
    """
    raise NotImplementedError("write logit_match_grad")


def feature_distill_loss(h_s: np.ndarray, h_t: np.ndarray, P: np.ndarray) -> float:
    """(1 / (2 n)) * ||h_s @ P - h_t||_F^2.

    Args:
        h_s: Student features, shape (n, d_s).
        h_t: Teacher features, shape (n, d_t).
        P: Projection from student to teacher width, shape (d_s, d_t).
    """
    raise NotImplementedError("write feature_distill_loss")


def feature_distill_grad(h_s: np.ndarray, h_t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Gradient of feature_distill_loss with respect to P, shape (d_s, d_t)."""
    raise NotImplementedError("write feature_distill_grad")


def fit_student(
    X: np.ndarray,
    z_t: np.ndarray,
    y: np.ndarray,
    T: float,
    alpha: float,
    steps: int = 400,
    lr: float = 0.1,
) -> np.ndarray:
    """Train a linear student z = X @ W by gradient descent on distill_loss.

    Args:
        X: Inputs, shape (n, d).
        z_t: Teacher logits on those inputs, shape (n, K).
        y: Hard labels, shape (n,).
        T, alpha: As in distill_loss.
        steps: Number of full-batch steps.
        lr: Step size.

    Returns:
        W of shape (d, K), started from zeros.
    """
    raise NotImplementedError("write fit_student")


if __name__ == "__main__":
    z_t = np.array([[6.0, 3.0, 0.0]])
    z_s = np.array([[4.0, 3.0, 1.0]])
    print("teacher at T=3:", soft_targets(z_t, 3.0).round(4))
    print("student at T=3:", soft_targets(z_s, 3.0).round(4))
    print("loss at T=3, alpha=0.5:", round(distill_loss(z_s, z_t, np.array([0]), 3.0, 0.5), 4))