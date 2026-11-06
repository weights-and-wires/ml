"""Learning-rate schedules: warmup and cosine decay.

Fill in `linear_warmup`, `cosine_decay_fraction`, and `cosine_schedule`.
`make_valley`, `adam_step`, and `run_with_schedule` are given.
"""

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
    """Return the rate during linear warmup.

    The rate rises in a straight line from 0 at step 0 to `peak` at step `warmup`,
    and stays at `peak` afterwards. When `warmup` is 0 there is no ramp, so the
    rate is `peak`.

    Args:
        step: The current step, 0-indexed.
        warmup: Number of warmup steps.
        peak: The rate at the end of warmup.

    Returns:
        The rate at `step`.

    Raises:
        ValueError: If `step` is negative.
    """
    raise NotImplementedError("linear_warmup")


def cosine_decay_fraction(p: float) -> float:
    """Return the cosine decay factor for progress `p` through the decay phase.

    Args:
        p: Progress, 0 at the start of the decay and 1 at its end. Values outside
            [0, 1] are clamped to that range.

    Returns:
        `0.5 * (1 + cos(pi * p))`, which is 1 at `p = 0`, 0.5 at `p = 0.5`, and
        0 at `p = 1`.
    """
    raise NotImplementedError("cosine_decay_fraction")


def cosine_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    """Return the rate at `step` under linear warmup and then cosine decay.

    For `step < warmup` the rate rises linearly to `peak`. From `warmup` to
    `total` it follows half a cosine from `peak` down to 0, and steps past
    `total` stay at 0. When `warmup >= total` there is no decay phase: the rate
    rises through warmup and stays at `peak`.

    Args:
        step: The current step, 0-indexed.
        warmup: Number of warmup steps.
        total: Total number of steps in the schedule.
        peak: The rate at the end of warmup.

    Returns:
        The rate at `step`.
    """
    raise NotImplementedError("cosine_schedule")


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
