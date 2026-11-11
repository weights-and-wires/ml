"""SGD, momentum, Adam, AdamW, and a cosine schedule.

Fill in the twelve functions marked `NotImplementedError`, in order. `make_valley`
and `run_with_schedule` are given. Every step function shares the signature

    new_w, new_state = step_fn(w, grad, state, **hyperparameters)

where `state` is a dict that carries memory from one step to the next. It is
empty on the first call.
"""

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
    """Return the gradient of half the mean squared miss over the rows in `idx`.

    Args:
        X: Inputs, shape (n, d).
        y: Targets, shape (n,).
        w: Weights, shape (d,).
        idx: Integer array of row numbers that form the mini-batch.

    Returns:
        The gradient, shape (d,). Divide by `len(idx)`, not by `n`.
    """
    raise NotImplementedError("batch_gradient")


def sgd_step(w: np.ndarray, grad: np.ndarray, state: dict, lr: float) -> tuple[np.ndarray, dict]:
    """Take one plain gradient step.

    Args:
        w: Current weights, any shape.
        grad: Gradient at `w`, same shape.
        state: Unused by SGD; return it unchanged.
        lr: Learning rate.

    Returns:
        A pair `(new_w, new_state)`.
    """
    raise NotImplementedError("sgd_step")


def momentum_step(
    w: np.ndarray, grad: np.ndarray, state: dict, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, dict]:
    """Take one heavy-ball step with velocity `v = beta * v + grad`.

    There is no `(1 - beta)` factor on the gradient, and the update is `w - lr * v`.

    Args:
        w: Current weights.
        grad: Gradient at `w`.
        state: Dict that may hold `"v"`, the velocity from the previous call.
            A missing `"v"` means zeros.
        lr: Learning rate.
        beta: Momentum coefficient in [0, 1).

    Returns:
        A pair `(new_w, new_state)`, where `new_state` holds the updated `"v"`.
    """
    raise NotImplementedError("momentum_step")


def ema_update(prev: np.ndarray, x: np.ndarray, beta: float) -> np.ndarray:
    """Blend a new value into a running average: `beta * prev + (1 - beta) * x`.

    Args:
        prev: The old average.
        x: The new value, same shape.
        beta: Decay in [0, 1). The old average keeps this weight.

    Returns:
        The new average, same shape as `prev`.
    """
    raise NotImplementedError("ema_update")


def bias_correct(avg: np.ndarray, beta: float, t: int) -> np.ndarray:
    """Divide out the share of the window that a zero-started average has not filled.

    Args:
        avg: A raw average that started at zero.
        beta: The decay that produced it.
        t: Number of updates so far, counted from 1.

    Returns:
        `avg / (1 - beta**t)`. Raises `ValueError` when `t < 1`.
    """
    raise NotImplementedError("bias_correct")


def adam_step(
    w: np.ndarray,
    grad: np.ndarray,
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Take one Adam step.

    Args:
        w: Current weights.
        grad: Gradient at `w`, same shape.
        state: Dict that may hold the raw averages `"m"` and `"v"` and the count
            `"t"`. Missing entries mean zeros and 0.
        lr: Learning rate.
        beta1: Decay of the gradient average.
        beta2: Decay of the squared-gradient average.
        eps: Small constant that keeps the denominator off zero.

    Returns:
        A pair `(new_w, new_state)`. The new state holds the raw `m`, the raw
        `v`, and `t`, never the corrected copies.
    """
    raise NotImplementedError("adam_step")


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
    """Take one Adam step on the raw gradient, then subtract `lr * weight_decay * w`.

    Use the `w` that came into the function in the decay term, not the new weights.
    The averages in the returned state never see the decay.

    Returns:
        A pair `(new_w, new_state)`.
    """
    raise NotImplementedError("adamw_step")


def linear_warmup(step: int, warmup: int, peak: float) -> float:
    """Return the rate at `step`: `peak * step / warmup` while `step < warmup`, then `peak`.

    A `warmup` of 0 means no warmup, so the rate is `peak` from step 0.
    Raises `ValueError` for a negative `step`.
    """
    raise NotImplementedError("linear_warmup")


def cosine_decay_fraction(p: float) -> float:
    """Return `0.5 * (1 + cos(pi * p))` after clamping `p` to [0, 1].

    The result is 1 at `p = 0`, 0.5 at `p = 0.5`, and 0 at `p = 1`.
    """
    raise NotImplementedError("cosine_decay_fraction")


def cosine_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    """Return the rate at `step`: linear warmup, then a cosine fall to 0 at `total`.

    `p` runs from 0 at the end of warmup to 1 at step `total`. When
    `warmup >= total` there is no decay phase: rise through warmup and hold `peak`.
    """
    raise NotImplementedError("cosine_schedule")


def minibatch_gradient_variance(per_example_grads: np.ndarray, batch_size: int) -> float:
    """Return the total variance of a mini-batch mean gradient, `tr(Sigma) / batch_size`.

    `Sigma` is the covariance matrix of the per-example gradients, taken over the
    rows you are given (divide by `n`, not `n - 1`). The total variance is the sum
    of the variances of the `d` coordinates, which is the trace of `Sigma`.

    Args:
        per_example_grads: One gradient per row, shape (n, d). A 1-D array of
            length n counts as d = 1.
        batch_size: Rows per mini-batch, drawn with replacement. Raises
            `ValueError` when it is below 1.

    Returns:
        The trace of the covariance of the batch mean, as a float.
    """
    raise NotImplementedError("minibatch_gradient_variance")


def gradient_noise_scale(per_example_grads: np.ndarray) -> float:
    """Return the simple gradient noise scale `tr(Sigma) / |G|^2`.

    `G` is the mean of the per-example gradients and `Sigma` is their covariance
    matrix, taken over the rows you are given (divide by `n`). It is the batch size
    at which the noise in a batch gradient equals the squared size of the signal.

    Args:
        per_example_grads: One gradient per row, shape (n, d). A 1-D array of
            length n counts as d = 1.

    Returns:
        A float. When `G` is exactly zero, return `float("inf")`.
    """
    raise NotImplementedError("gradient_noise_scale")


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
