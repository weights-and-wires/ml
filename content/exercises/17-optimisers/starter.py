"""Four optimisers behind one interface.

Every step function shares the signature

    new_w, new_state = step_fn(w, grad, state, **hyperparameters)

where `state` is a dict holding whatever accumulators the optimiser needs
between calls (empty for a fresh run). That shared shape is what lets
`run_optimiser` drive any of them with the same loop.

Fill in the six functions below. `w` and `grad` are numpy arrays of matching
shape; every optimiser must work elementwise, so it makes no difference
whether that shape is `(1,)` or `(50,)`.
"""

import numpy as np


def sgd_step(w: np.ndarray, grad: np.ndarray, state: dict, lr: float) -> tuple[np.ndarray, dict]:
    """Take one plain gradient-descent step.

    Args:
        w: Current parameters, any shape.
        grad: Gradient at `w`, same shape as `w`.
        state: Unused by SGD; pass through unchanged.
        lr: Learning rate.

    Returns:
        A tuple `(new_w, new_state)`.
    """
    raise NotImplementedError("sgd_step")


def momentum_step(
    w: np.ndarray, grad: np.ndarray, state: dict, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, dict]:
    """Take one heavy-ball momentum step.

    The velocity recursion is `v = beta * v + grad`, with no normalising
    `(1 - beta)` factor, and the update is `w - lr * v`.

    Args:
        w: Current parameters.
        grad: Gradient at `w`.
        state: Dict that may hold `"v"`, the velocity from the previous call.
            Treat a missing `"v"` as zeros.
        lr: Learning rate.
        beta: Momentum coefficient in `[0, 1)`.

    Returns:
        A tuple `(new_w, new_state)`, where `new_state` holds the updated `"v"`.
    """
    raise NotImplementedError("momentum_step")


def adam_step(
    w: np.ndarray,
    grad: np.ndarray,
    state: dict,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Take one Adam step, with bias-corrected first and second moments.

    Maintain `m = beta1 * m + (1 - beta1) * grad` and
    `v = beta2 * v + (1 - beta2) * grad**2`, correct both by dividing by
    `1 - beta1**t` and `1 - beta2**t` respectively, where `t` is the number of
    updates seen so far including this one, then update
    `w - lr * m_hat / (sqrt(v_hat) + eps)`.

    Args:
        w: Current parameters.
        grad: Gradient at `w`.
        state: Dict that may hold `"m"`, `"v"`, and `"t"` from the previous
            call. Treat missing `"m"` and `"v"` as zeros and missing `"t"` as 0.
        lr: Learning rate.
        beta1: First-moment decay.
        beta2: Second-moment decay.
        eps: Numerical-stability constant in the denominator.

    Returns:
        A tuple `(new_w, new_state)`, where `new_state` holds the updated
        `"m"`, `"v"`, and `"t"`.
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
    """Take one AdamW step: Adam's moment update, plus decoupled weight decay.

    Compute the moments from `grad` alone, exactly as `adam_step` does, then
    subtract `lr * weight_decay * w` (the pre-update `w`) from the result. Do
    not add `weight_decay * w` to the gradient before computing the moments.

    Args:
        w: Current parameters.
        grad: Gradient at `w`, without any weight-decay term folded in.
        state: Same contract as `adam_step`.
        lr: Learning rate.
        beta1: First-moment decay.
        beta2: Second-moment decay.
        eps: Numerical-stability constant in the denominator.
        weight_decay: Decoupled decay rate, applied outside the adaptive step.

    Returns:
        A tuple `(new_w, new_state)`.
    """
    raise NotImplementedError("adamw_step")


def cosine_schedule(step: int, warmup: int, total: int, peak: float) -> float:
    """Return the learning rate at `step` under linear warmup and cosine decay.

    For `step < warmup`, the rate rises linearly from 0 to `peak`. For
    `warmup <= step <= total`, it follows a cosine curve from `peak` down to 0.
    When `warmup >= total`, return `peak` for every step at or past warmup.

    Args:
        step: The current step, 0-indexed.
        warmup: Number of warmup steps.
        total: Total number of steps in the schedule.
        peak: The learning rate at the end of warmup.

    Returns:
        The learning rate at `step`, as a float.
    """
    raise NotImplementedError("cosine_schedule")


def run_optimiser(grad_fn, init: np.ndarray, step_fn, steps: int):
    """Run `step_fn` for `steps` iterations starting from `init`.

    Args:
        grad_fn: A callable mapping the current `w` to its gradient.
        init: The starting point, any shape.
        step_fn: A callable `(w, grad, state) -> (new_w, new_state)`, for
            example `functools.partial(adam_step, lr=0.01)`.
        steps: Number of iterations to run.

    Returns:
        A tuple `(final_w, history)`, where `history` is a list of the
        iterates visited, starting with `init` and holding `steps + 1`
        entries in total.
    """
    raise NotImplementedError("run_optimiser")


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
