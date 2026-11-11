"""Reference solution for the distillation exercise."""

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
    a = np.asarray(logits, dtype=float) / T
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) = sum_k p_k log(p_k / q_k) along the last axis.

    Args:
        p: Probabilities of shape (K,) or (n, K). A zero entry contributes 0.
        q: Probabilities of the same shape, positive wherever p is positive.

    Returns:
        Array of shape () or (n,), one divergence per row.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    safe_p = np.where(p > 0, p, 1.0)
    safe_q = np.where(p > 0, q, 1.0)
    return np.sum(p * (np.log(safe_p) - np.log(safe_q)), axis=-1)


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
    z_s = np.asarray(z_s, dtype=float)
    n = len(z_s)
    soft = kl_divergence(soft_targets(z_t, T), soft_targets(z_s, T))
    p1 = soft_targets(z_s, 1.0)
    hard = -np.log(p1[np.arange(n), y])
    return float(np.mean(alpha * T**2 * soft + (1 - alpha) * hard))


def distill_grad(z_s: np.ndarray, z_t: np.ndarray, y: np.ndarray, T: float, alpha: float) -> np.ndarray:
    """Gradient of distill_loss with respect to z_s, shape (n, K)."""
    z_s = np.asarray(z_s, dtype=float)
    n, K = z_s.shape
    soft = T * (soft_targets(z_s, T) - soft_targets(z_t, T))
    hard = soft_targets(z_s, 1.0) - np.eye(K)[y]
    return (alpha * soft + (1 - alpha) * hard) / n


def logit_match_grad(z_s: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    """Limit of the gradient of T^2 * KL(p_t^T || p_s^T) as T grows: (z_s_c - z_t_c) / K.

    Args:
        z_s: Student logits, shape (K,) or (n, K).
        z_t: Teacher logits, the same shape.

    Returns:
        Array of the same shape. The subscript c means centred: subtract the row mean.
    """
    z_s = np.asarray(z_s, dtype=float)
    z_t = np.asarray(z_t, dtype=float)
    K = z_s.shape[-1]
    zs_c = z_s - z_s.mean(axis=-1, keepdims=True)
    zt_c = z_t - z_t.mean(axis=-1, keepdims=True)
    return (zs_c - zt_c) / K


def feature_distill_loss(h_s: np.ndarray, h_t: np.ndarray, P: np.ndarray) -> float:
    """(1 / (2 n)) * ||h_s @ P - h_t||_F^2.

    Args:
        h_s: Student features, shape (n, d_s).
        h_t: Teacher features, shape (n, d_t).
        P: Projection from student to teacher width, shape (d_s, d_t).
    """
    r = np.asarray(h_s) @ np.asarray(P) - np.asarray(h_t)
    return float(np.sum(r**2) / (2 * len(r)))


def feature_distill_grad(h_s: np.ndarray, h_t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Gradient of feature_distill_loss with respect to P, shape (d_s, d_t)."""
    h_s = np.asarray(h_s)
    r = h_s @ np.asarray(P) - np.asarray(h_t)
    return h_s.T @ r / len(r)


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
    W = np.zeros((X.shape[1], z_t.shape[1]))
    for _ in range(steps):
        W = W - lr * X.T @ distill_grad(X @ W, z_t, y, T, alpha)
    return W


if __name__ == "__main__":
    z_t = np.array([[6.0, 3.0, 0.0]])
    z_s = np.array([[4.0, 3.0, 1.0]])
    print("teacher at T=3:", soft_targets(z_t, 3.0).round(4))
    print("student at T=3:", soft_targets(z_s, 3.0).round(4))
    print("loss at T=3, alpha=0.5:", round(distill_loss(z_s, z_t, np.array([0]), 3.0, 0.5), 4))
