"""Decay the weights outside Adam.

Fill in `adam_l2_step`, `adamw_step`, and `decay_amount`. `sgd_step` and
`adam_step` are given from earlier chapters. Every step function shares the
signature

    new_w, new_state = step_fn(w, grad, state, **hyperparameters)
"""

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
    """Take an Adam step where the L2 penalty is added to the gradient.

    The gradient that reaches the averages is `grad + weight_decay * w`, so the
    decay passes through the moving averages and Adam's division.

    Args:
        w: Current weights.
        grad: Gradient of the loss alone, without any penalty.
        state: Same contract as `adam_step`.
        lr: Learning rate.
        beta1: Decay for the average gradient.
        beta2: Decay for the average squared gradient.
        eps: Constant added to the denominator.
        weight_decay: L2 penalty strength.

    Returns:
        A pair `(new_w, new_state)`.
    """
    raise NotImplementedError("adam_l2_step")


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
    """Take an AdamW step: Adam on the raw gradient, then decoupled decay.

    The averages see only `grad`. After the Adam step, subtract
    `lr * weight_decay * w`, using the `w` that came into this call.

    Args:
        w: Current weights.
        grad: Gradient of the loss alone.
        state: Same contract as `adam_step`.
        lr: Learning rate.
        beta1: Decay for the average gradient.
        beta2: Decay for the average squared gradient.
        eps: Constant added to the denominator.
        weight_decay: Decay rate.

    Returns:
        A pair `(new_w, new_state)`. The state is the one Adam would return.
    """
    raise NotImplementedError("adamw_step")


def decay_amount(step_fn, w: np.ndarray, grads: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    """Measure the shrink that decay alone causes in one step.

    Hold `w` fixed. Run `step_fn(w, g, state, lr=lr, weight_decay=d)` over every
    row `g` of `grads`, passing the returned state to the next call, and keep
    the weights returned by the last call. Do this once with `d = 0` and once
    with `d = weight_decay`.

    Args:
        step_fn: A step function that accepts `lr` and `weight_decay` keywords.
        w: Weights, shape (d,), held fixed during the run.
        grads: Gradients, shape (T, d), fed in order.
        lr: Learning rate.
        weight_decay: Decay rate for the second run.

    Returns:
        The last weights without decay minus the last weights with decay,
        shape (d,).
    """
    raise NotImplementedError("decay_amount")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(200, 2)) * np.array([0.01, 10.0])  # a quiet weight and a noisy weight
    w = np.ones(2)
    for name, fn in [("L2 inside Adam", adam_l2_step), ("AdamW", adamw_step)]:
        shrink = decay_amount(fn, w, grads, lr=0.1, weight_decay=0.01)
        print(f"{name:>15}: shrink from decay alone = {shrink} (intended 0.001 each)")
