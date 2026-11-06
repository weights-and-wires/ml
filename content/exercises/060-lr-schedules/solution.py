"""Reference solution for the learning-rate schedules exercise."""

import numpy as np


# Given: written in the chapter "Noisy gradients: stochastic gradient descent and mini-batches".
def make_valley(n: int = 200, seed: int = 0, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    X = np.sqrt(n) * q * np.array([1.0, 10.0])  # X.T @ X / n is diag(1, 100)
    y = X @ np.array([1.0, 0.1]) + noise * rng.normal(size=n)
    return X, y


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


# Given: the loop for this chapter.
def run_with_schedule(grad_fn, init: np.ndarray, step_fn, steps: int, schedule_fn):
    """Run `step_fn` for `steps` iterations, using `schedule_fn(k)` as the rate at step `k`.

    Returns `(final_w, history, lrs)`, where `history` has `steps + 1` entries and
    `lrs` holds the `steps` rates that were used.
    """
    w = np.array(init, dtype=float)
    state: dict = {}
    history = [w.copy()]
    lrs = []
    for k in range(steps):
        lr = schedule_fn(k)
        lrs.append(lr)
        w, state = step_fn(w, grad_fn(w), state, lr=lr)
        history.append(w.copy())
    return w, history, lrs


def linear_warmup(step: int, warmup: int, peak: float) -> float:
    if step < 0:
        raise ValueError(f"step must be 0 or more, got {step}")
    if warmup <= 0 or step >= warmup:
        return peak
    return peak * step / warmup


def cosine_decay_fraction(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * p))


def cosine_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup or warmup >= total:
        return linear_warmup(step, warmup, peak)
    p = (step - warmup) / (total - warmup)
    return peak * cosine_decay_fraction(p)


if __name__ == "__main__":
    import functools

    for step in (0, 50, 100, 325, 550, 1000):
        print(f"cosine_schedule({step:>4}) = {cosine_schedule(step, 100, 1000, 0.01):.7f}")

    X, y = make_valley()
    n = len(y)
    best = np.linalg.solve(X.T @ X, X.T @ y)
    excess = lambda w: 0.5 * np.mean((X @ w - y) ** 2) - 0.5 * np.mean((X @ best - y) ** 2)
    steps, peak = 400, 0.1
    schedules = {
        "constant": lambda k: peak,
        "warmup + cosine": functools.partial(cosine_schedule, warmup=20, total=steps, peak=peak),
    }
    for name, schedule_fn in schedules.items():
        tails = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            grad_fn = lambda w: (lambda i: X[i].T @ (X[i] @ w - y[i]) / len(i))(rng.choice(n, 8, replace=False))
            _, history, _ = run_with_schedule(grad_fn, np.array([-1.0, 0.3]), adam_step, steps, schedule_fn)
            tails.append(np.mean([excess(w) for w in history[-50:]]))
        print(f"{name:>16}: excess loss over the last 50 steps = {np.mean(tails):.4f}")
