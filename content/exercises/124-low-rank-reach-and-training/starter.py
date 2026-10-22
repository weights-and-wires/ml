"""Truncate an update and train its factors.

Fill in the three functions below. `merge_lora` is already written: it is the
function you wrote in the chapter "Merging an adapter back into the weight".
"""

import numpy as np


# Written in the chapter "Merging an adapter back into the weight".
def merge_lora(
    W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    return W + (alpha / r) * (B @ A)


def best_rank_approximation(delta_w: np.ndarray, r: int) -> np.ndarray:
    """Return the closest rank-r matrix to delta_w in Frobenius norm.

    Args:
        delta_w: A matrix, shape (d_out, d_in).
        r: The rank to keep, between 1 and min(delta_w.shape).

    Returns:
        The matrix keeping the r largest singular values, same shape as delta_w.
    """
    raise NotImplementedError("best_rank_approximation")


def energy_captured(delta_w: np.ndarray, r: int) -> float:
    """Return the share of squared singular values kept at rank r.

    Args:
        delta_w: A nonzero matrix, shape (d_out, d_in).
        r: The rank to keep, between 0 and min(delta_w.shape).

    Returns:
        (s_1^2 + ... + s_r^2) / (sum of all s_i^2), a float between 0 and 1.
    """
    raise NotImplementedError("energy_captured")


def lora_gradients(
    W: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
    r: int,
    target: np.ndarray,
) -> tuple:
    """Return the gradients of 0.5 * ||W_eff - target||^2 with respect to A and B.

    W_eff is merge_lora(W, A, B, alpha, r), and ||.|| is the Frobenius norm.

    Args:
        W: Frozen weight, shape (d_out, d_in).
        A: Low-rank factor, shape (r, d_in).
        B: Low-rank factor, shape (d_out, r).
        alpha: The scale numerator.
        r: The rank.
        target: The weight to match, shape (d_out, d_in).

    Returns:
        (dA, dB) with the shapes of A and B.
    """
    raise NotImplementedError("lora_gradients")


if __name__ == "__main__":
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    approx = best_rank_approximation(M, 1)
    print("best rank-1 approximation of [[2, 1], [1, 2]]:")
    print(approx)
    print("error:", np.linalg.norm(M - approx), " energy kept:", energy_captured(M, 1))

    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    target = np.array([[1.0, 1.0], [2.0, 0.0]])
    dA, dB = lora_gradients(W, A, B, 2.0, 1, target)
    print("dA (expect [[4, 6]]):", dA)
    print("dB (expect [[16], [0]]):", dB.ravel())

    eta = 0.01
    before = merge_lora(W, A, B, 2.0, 1)
    after = merge_lora(W, A - eta * dA, B - eta * dB, 2.0, 1)
    print("change in W_eff after one step:")
    print(after - before)
