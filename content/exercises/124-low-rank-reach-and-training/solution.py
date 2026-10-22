"""Reference solution for the low-rank reach and training exercise."""

import numpy as np


# Written in the chapter "Merging an adapter back into the weight".
def merge_lora(
    W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int
) -> np.ndarray:
    return W + (alpha / r) * (B @ A)


def best_rank_approximation(delta_w: np.ndarray, r: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(delta_w, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def energy_captured(delta_w: np.ndarray, r: int) -> float:
    s = np.linalg.svd(delta_w, compute_uv=False)
    return float((s[:r] ** 2).sum() / (s**2).sum())


def lora_gradients(
    W: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
    r: int,
    target: np.ndarray,
) -> tuple:
    G = merge_lora(W, A, B, alpha, r) - target
    scale = alpha / r
    dA = scale * (B.T @ G)
    dB = scale * (G @ A.T)
    return dA, dB


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
