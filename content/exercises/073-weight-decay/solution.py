"""Reference solution for the weight-decay exercise."""

import numpy as np


def l2_penalty_grad(w: np.ndarray, lam: float) -> np.ndarray:
    return lam * w


def weight_decay_step(w: np.ndarray, grad: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    return w - lr * (grad + l2_penalty_grad(w, weight_decay))


def step_angle(w: np.ndarray, grad: np.ndarray, lr: float) -> float:
    return float(np.arctan(lr * np.linalg.norm(grad) / np.linalg.norm(w)))


if __name__ == "__main__":
    w = np.array([1.0])
    print("worked step:", weight_decay_step(w, np.array([0.1]), lr=0.1, weight_decay=0.01))

    w = np.array([1.0])
    for _ in range(30):
        w = weight_decay_step(w, np.zeros(1), lr=0.5, weight_decay=0.2)
    print(f"no data gradient, 30 steps: {w[0]:.4f} (0.9**30 = {0.9 ** 30:.4f})")

    w = np.array([1.0])
    for _ in range(200):
        w = weight_decay_step(w, w - 0.6, lr=0.5, weight_decay=0.2)
    print(f"data pulls to 0.6, decay 0.2: settles at {w[0]:.4f}")

    for length in (4.0, 2.0):
        angle = step_angle(np.array([length, 0.0]), np.array([0.0, 1.0]), lr=1.0)
        print(f"|w| = {length}: one sideways step turns it {np.degrees(angle):.1f} degrees")
