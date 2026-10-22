"""Reference solution for the AdamW exercise."""

import numpy as np


# Given: written in the chapter "Noisy gradients: stochastic gradient descent and mini-batches".
def sgd_step(w: np.ndarray, grad: np.ndarray, state: dict, lr: float) -> tuple[np.ndarray, dict]:
    return w - lr * grad, state


# Given: written in the chapter "Adam: a step size for every weight".
def adam_step(
    w: np.ndarray,
    grad: np.ndarray,
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    t = state.get("t", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(w)) + (1.0 - beta1) * grad
    v = beta2 * state.get("v", np.zeros_like(w)) + (1.0 - beta2) * grad**2
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps), {"m": m, "v": v, "t": t}


def adam_l2_step(
    w: np.ndarray,
    grad: np.ndarray,
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, dict]:
    return adam_step(w, grad + weight_decay * w, state, lr, beta1, beta2, eps)


def adamw_step(
    w: np.ndarray,
    grad: np.ndarray,
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, dict]:
    new_w, new_state = adam_step(w, grad, state, lr, beta1, beta2, eps)
    return new_w - lr * weight_decay * w, new_state


def decay_amount(step_fn, w: np.ndarray, grads: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    last = []
    for decay in (0.0, weight_decay):
        state: dict = {}
        for g in grads[:-1]:
            _, state = step_fn(w, g, state, lr=lr, weight_decay=decay)
        new_w, _ = step_fn(w, grads[-1], state, lr=lr, weight_decay=decay)
        last.append(new_w)
    return last[0] - last[1]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(200, 2)) * np.array([0.01, 10.0])  # a quiet weight and a noisy weight
    w = np.ones(2)
    for name, fn in [("L2 inside Adam", adam_l2_step), ("AdamW", adamw_step)]:
        shrink = decay_amount(fn, w, grads, lr=0.1, weight_decay=0.01)
        print(f"{name:>15}: shrink from decay alone = {shrink} (intended 0.001 each)")
