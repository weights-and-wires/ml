"""Reference solution for the optimisers exercise."""

import numpy as np


def sgd_step(w: np.ndarray, grad: np.ndarray, state: dict, lr: float) -> tuple[np.ndarray, dict]:
    return w - lr * grad, state


def momentum_step(
    w: np.ndarray, grad: np.ndarray, state: dict, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, dict]:
    v_prev = state.get("v", np.zeros_like(w))
    v = beta * v_prev + grad
    new_w = w - lr * v
    return new_w, {"v": v}


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
    m_prev = state.get("m", np.zeros_like(w))
    v_prev = state.get("v", np.zeros_like(w))
    m = beta1 * m_prev + (1 - beta1) * grad
    v = beta2 * v_prev + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    new_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_w, {"m": m, "v": v, "t": t}


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
    new_w = new_w - lr * weight_decay * w
    return new_w, new_state


def cosine_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    if warmup >= total:
        return peak
    if step < warmup:
        return peak * step / warmup
    progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))


def run_optimiser(grad_fn, init: np.ndarray, step_fn, steps: int):
    w = np.array(init, dtype=float)
    state: dict = {}
    history = [w.copy()]
    for _ in range(steps):
        grad = grad_fn(w)
        w, state = step_fn(w, grad, state)
        history.append(w.copy())
    return w, history


if __name__ == "__main__":
    import functools

    a = np.array([1.0, 100.0])
    grad_fn = lambda w: a * w
    loss_fn = lambda w: 0.5 * np.sum(a * w**2)
    init = np.array([1.0, 1.0])

    for name, fn, kwargs, steps in [
        ("sgd", sgd_step, {"lr": 0.0095}, 4000),
        ("momentum", momentum_step, {"lr": 0.004, "beta": 0.9}, 2000),
        ("adam", adam_step, {"lr": 0.05}, 1500),
        ("adamw", adamw_step, {"lr": 0.05, "weight_decay": 0.0}, 1500),
    ]:
        step_fn = functools.partial(fn, **kwargs)
        w_final, history = run_optimiser(grad_fn, init.copy(), step_fn, steps)
        print(f"{name:>9}: final loss {loss_fn(w_final):.3e} after {steps} steps")

    for step in (0, 50, 100, 500, 1000):
        print(f"cosine_schedule({step}) = {cosine_schedule(step, warmup=100, total=1000, peak=0.01):.5f}")
