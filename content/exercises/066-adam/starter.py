"""Adam from moving averages.

Fill in `ema_update`, `bias_correct`, and `adam_step`. The functions marked
"Given" come from the previous chapter or help you compare optimisers.
Every step function shares the signature

    new_w, new_state = step_fn(w, grad, state, **hyperparameters)

where `state` is a dict that carries memory from one step to the next.
"""

import numpy as np


# Given: written in the chapter "Noisy gradients: stochastic gradient descent and mini-batches".
def sgd_step(w: np.ndarray, grad: np.ndarray, state: dict, lr: float) -> tuple[np.ndarray, dict]:
    return w - lr * grad, state


# Given: written in the chapter "Noisy gradients: stochastic gradient descent and mini-batches".
def momentum_step(
    w: np.ndarray, grad: np.ndarray, state: dict, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, dict]:
    v = beta * state.get("v", np.zeros_like(w)) + grad
    return w - lr * v, {"v": v}


# Given: the loop from the chapter "Noisy gradients: stochastic gradient descent and mini-batches".
def run_optimiser(grad_fn, init: np.ndarray, step_fn, steps: int):
    """Call grad_fn and step_fn in a loop; return (final_w, history)."""
    w = np.array(init, dtype=float)
    state: dict = {}
    history = [w.copy()]
    for _ in range(steps):
        w, state = step_fn(w, grad_fn(w), state)
        history.append(w.copy())
    return w, history


# Given: a helper for comparing optimisers.
def steps_to_settle(step_fn, grad_fn, loss_fn, init: np.ndarray, target: float, max_steps: int) -> int:
    """Return the step from which the loss stays below `target` through `max_steps`.

    Returns `max_steps + 1` when the loss is still at or above `target` at the end.
    """
    _, history = run_optimiser(grad_fn, init, step_fn, max_steps)
    above = [t for t, w in enumerate(history) if loss_fn(w) >= target]
    return above[-1] + 1 if above else 0


def ema_update(prev: np.ndarray, x: np.ndarray, beta: float) -> np.ndarray:
    """Blend a new value into a moving average.

    Args:
        prev: The old average, any shape.
        x: The new value, same shape.
        beta: Decay in [0, 1). The old average keeps weight `beta`.

    Returns:
        The updated average, `beta * prev + (1 - beta) * x`.
    """
    raise NotImplementedError("ema_update")


def bias_correct(avg: np.ndarray, beta: float, t: int) -> np.ndarray:
    """Undo the shrinkage of an average that started at zero.

    Args:
        avg: The raw moving average.
        beta: The decay used to build it.
        t: Number of updates so far, counted from 1.

    Returns:
        `avg / (1 - beta**t)`.

    Raises:
        ValueError: If `t < 1`.
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
    """Take one Adam step with bias-corrected first and second moments.

    Args:
        w: Current weights.
        grad: Gradient at `w`, same shape.
        state: Dict that may hold the raw `"m"`, the raw `"v"`, and `"t"`.
            Missing `"m"` and `"v"` mean zeros, and a missing `"t"` means 0.
        lr: Learning rate.
        beta1: Decay for the average gradient.
        beta2: Decay for the average squared gradient.
        eps: Constant added to the denominator.

    Returns:
        A pair `(new_w, new_state)`. The new state holds the raw `"m"`, the
        raw `"v"`, and the step count `"t"`, never the corrected copies.
    """
    raise NotImplementedError("adam_step")


if __name__ == "__main__":
    import functools

    # The hot weight and the cold weight from the chapter.
    for g in (3.0, 3000.0):
        w, _ = adam_step(np.zeros(1), np.array([g]), {}, lr=0.1)
        print(f"gradient {g:>6}: Adam's first step is {-w[0]:.4f}")

    init = np.array([1.0, 1.0])
    for steepness in (100.0, 10000.0):
        a = np.array([1.0, steepness])
        grad_fn = lambda w: a * w
        loss_fn = lambda w: 0.5 * float(np.sum(a * w**2))
        print(f"curvatures 1 and {steepness:g}")
        for name, step_fn in [
            ("sgd", functools.partial(sgd_step, lr=1.95 / steepness)),
            ("momentum", functools.partial(momentum_step, lr=1.6 / steepness, beta=0.9)),
            ("adam", functools.partial(adam_step, lr=0.05)),
        ]:
            n = steps_to_settle(step_fn, grad_fn, loss_fn, init, 1e-4, 30000 if name != "adam" else 3000)
            print(f"{name:>9}: loss stays below 1e-4 from step {n}")
