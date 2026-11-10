"""Reference solution: SGD, momentum, Adam, AdamW, and a cosine schedule."""

import numpy as np


def make_valley(n: int = 200, seed: int = 0, noise: float = 1.0):
    """Given. Build a regression problem whose loss has curvatures 1 and 100."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    X = np.sqrt(n) * q * np.array([1.0, 10.0])  # X.T @ X / n is diag(1, 100)
    y = X @ np.array([1.0, 0.1]) + noise * rng.normal(size=n)
    return X, y


def run_with_schedule(grad_fn, init: np.ndarray, step_fn, steps: int, schedule_fn):
    """Given. Run `step_fn` for `steps` iterations, using `schedule_fn(k)` as the rate at step `k`.

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


def batch_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    Xb, yb = X[idx], y[idx]
    return Xb.T @ (Xb @ w - yb) / len(idx)


def sgd_step(w: np.ndarray, grad: np.ndarray, state: dict, lr: float) -> tuple[np.ndarray, dict]:
    return w - lr * grad, state


def momentum_step(
    w: np.ndarray, grad: np.ndarray, state: dict, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, dict]:
    v = beta * state.get("v", np.zeros_like(w)) + grad
    return w - lr * v, {"v": v}


def ema_update(prev: np.ndarray, x: np.ndarray, beta: float) -> np.ndarray:
    return beta * prev + (1.0 - beta) * x


def bias_correct(avg: np.ndarray, beta: float, t: int) -> np.ndarray:
    if t < 1:
        raise ValueError(f"t counts updates from 1, got {t}")
    return avg / (1.0 - beta**t)


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
    m = ema_update(state.get("m", np.zeros_like(w)), grad, beta1)
    v = ema_update(state.get("v", np.zeros_like(w)), grad**2, beta2)
    m_hat = bias_correct(m, beta1, t)
    v_hat = bias_correct(v, beta2, t)
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
    return new_w - lr * weight_decay * w, new_state


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


def _as_rows(per_example_grads):
    g = np.asarray(per_example_grads, dtype=float)
    return g.reshape(len(g), -1)


def minibatch_gradient_variance(per_example_grads: np.ndarray, batch_size: int) -> float:
    """Return the total variance of a mini-batch mean gradient, `tr(Sigma) / batch_size`."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    g = _as_rows(per_example_grads)
    return float(g.var(axis=0).sum() / batch_size)


def gradient_noise_scale(per_example_grads: np.ndarray) -> float:
    """Return the simple gradient noise scale `tr(Sigma) / |G|^2`."""
    g = _as_rows(per_example_grads)
    signal = float(np.sum(g.mean(axis=0) ** 2))
    if signal == 0.0:
        return float("inf")
    return float(g.var(axis=0).sum() / signal)


if __name__ == "__main__":
    import functools

    X, y = make_valley()
    n = len(y)
    best = np.linalg.solve(X.T @ X, X.T @ y)
    excess = lambda w: 0.5 * np.mean((X @ w - y) ** 2) - 0.5 * np.mean((X @ best - y) ** 2)
    init = np.array([-1.0, 0.3])

    def noisy_run(step_fn, steps, schedule_fn=None, seed=1):
        """Run `step_fn` on batches of eight rows and return the excess loss at every step."""
        rng = np.random.default_rng(seed)
        grad_fn = lambda w: batch_gradient(X, y, w, rng.choice(n, 8, replace=False))
        w, state, losses = init.copy(), {}, []
        for k in range(steps):
            kwargs = {} if schedule_fn is None else {"lr": schedule_fn(k)}
            w, state = step_fn(w, grad_fn(w), state, **kwargs)
            losses.append(excess(w))
        return np.array(losses)

    # 1. Mini-batch SGD against momentum.
    for name, step_fn in [
        ("sgd", functools.partial(sgd_step, lr=0.0095)),
        ("momentum", functools.partial(momentum_step, lr=0.0025, beta=0.9)),
    ]:
        losses = noisy_run(step_fn, 300)
        reached = int(np.argmax(losses < 0.5)) if np.any(losses < 0.5) else None
        print(f"{name:>8}: excess loss below 0.5 at step {reached}, final {losses[-1]:.3f}")

    # 2. Adam's first step for a gradient of 3 and of 3000.
    for g in (3.0, 3000.0):
        w, _ = adam_step(np.zeros(1), np.array([g]), {}, lr=0.1)
        print(f"gradient {g:>6}: Adam's first step is {-w[0]:.4f}")

    # 3. Shrink from decay alone, for a quiet weight and a noisy weight.
    grads = np.random.default_rng(0).normal(size=(200, 2)) * np.array([0.01, 10.0])
    w0 = np.ones(2)

    def last_step(step_fn, decay):
        state: dict = {}
        for g in grads[:-1]:
            _, state = step_fn(w0, g, state, lr=0.1, weight_decay=decay)
        return step_fn(w0, grads[-1], state, lr=0.1, weight_decay=decay)[0]

    l2 = lambda w, g, s, lr, weight_decay: adam_step(w, g + weight_decay * w, s, lr)
    for name, fn in [("L2 inside Adam", l2), ("AdamW", adamw_step)]:
        shrink = last_step(fn, 0.0) - last_step(fn, 0.01)
        print(f"{name:>15}: shrink from decay alone = {shrink} (intended 0.001 each)")

    # 4. The schedule, then Adam with and without it.
    for step in (0, 50, 100, 325, 550, 1000):
        print(f"cosine_schedule({step:>4}) = {cosine_schedule(step, 100, 1000, 0.01):.7f}")
    steps, peak = 400, 0.1
    constant = noisy_run(adam_step, steps, lambda k: peak)
    scheduled = noisy_run(adam_step, steps, functools.partial(cosine_schedule, warmup=20, total=steps, peak=peak))
    print(f"Adam, last 50 steps: constant {constant[-50:].mean():.4f}, scheduled {scheduled[-50:].mean():.4f}")

    # 5. Noise scale of the per-row gradients at the start and near the bottom.
    for label, w in [("start", init), ("near the bottom", best + np.array([0.01, 0.001]))]:
        per_row = X * (X @ w - y)[:, None]
        print(f"gradient noise scale {label:>15}: {gradient_noise_scale(per_row):,.1f}")
