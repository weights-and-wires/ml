"""Reference solution for the weight-decay and early-stopping exercise."""

import numpy as np


def l2_penalty_grad(w: np.ndarray, lam: float) -> np.ndarray:
    return lam * w


def weight_decay_step(w: np.ndarray, grad: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    return w - lr * (grad + l2_penalty_grad(w, weight_decay))


def step_angle(w: np.ndarray, grad: np.ndarray, lr: float) -> float:
    return float(np.arctan(lr * np.linalg.norm(grad) / np.linalg.norm(w)))


def gd_quadratic_1d(eigenvalue: float, step: float, w_star: float, iterations: int) -> float:
    w = 0.0
    for _ in range(iterations):
        w = w - step * eigenvalue * (w - w_star)
    return float(w)


def early_stopping_equivalent_ridge(eigenvalue: float, step: float, iterations: int) -> float:
    contraction = step * eigenvalue
    if not 0.0 < contraction <= 1.0 + 1e-12:
        raise ValueError(
            f"step * eigenvalue must be in (0, 1], got {contraction}: "
            "outside that range no positive ridge penalty matches gradient descent"
        )
    if iterations == 0:
        return float("inf")
    remaining = (1.0 - contraction) ** iterations
    return float(eigenvalue * remaining / (1.0 - remaining))


def best_stopping_step(val_losses, patience: int) -> int:
    best_index = 0
    best_loss = val_losses[0]
    since_best = 0
    for i in range(1, len(val_losses)):
        if val_losses[i] < best_loss:
            best_index, best_loss, since_best = i, val_losses[i], 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    return best_index


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

    c, a, w_star = 2.0, 0.1, 1.0
    print("k   gradient descent   ridge solution   penalty")
    for k in (0, 1, 10, 50):
        w = gd_quadratic_1d(c, a, w_star, k)
        lam = early_stopping_equivalent_ridge(c, a, k)
        ridge = 0.0 if np.isinf(lam) else c / (c + lam) * w_star
        print(f"{k:<3} {w:>16.5f} {ridge:>16.5f} {lam:>10.5f}")

    losses = [5.0, 4.0, 4.5, 3.0, 3.5, 3.6, 3.7]
    for patience in (1, 2):
        print(f"patience {patience}: stop at step {best_stopping_step(losses, patience)}")
