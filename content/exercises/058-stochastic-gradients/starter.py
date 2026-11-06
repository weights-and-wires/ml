"""Noisy gradients and a heavy ball.

Fill in the three functions marked `NotImplementedError`. `make_valley` and
`run_optimiser` are given. Every step function shares the signature

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


def run_optimiser(grad_fn, init: np.ndarray, step_fn, steps: int):
    """Given. Call grad_fn and step_fn in a loop; return (final_w, history)."""
    w = np.array(init, dtype=float)
    state: dict = {}
    history = [w.copy()]
    for _ in range(steps):
        w, state = step_fn(w, grad_fn(w), state)
        history.append(w.copy())
    return w, history


if __name__ == "__main__":
    import functools

    X, y = make_valley()
    n = len(y)
    w_best = np.linalg.solve(X.T @ X, X.T @ y)
    excess = lambda w: 0.5 * np.mean((X @ w - y) ** 2) - 0.5 * np.mean((X @ w_best - y) ** 2)
    init = np.array([-1.0, 0.3])

    for name, step_fn in [
        ("sgd", functools.partial(sgd_step, lr=0.0095)),
        ("momentum", functools.partial(momentum_step, lr=0.0025, beta=0.9)),
    ]:
        rng = np.random.default_rng(1)
        grad_fn = lambda w: batch_gradient(X, y, w, rng.choice(n, 8, replace=False))
        _, history = run_optimiser(grad_fn, init, step_fn, 300)
        losses = np.array([excess(w) for w in history])
        reached = int(np.argmax(losses < 0.5)) if np.any(losses < 0.5) else None
        print(f"{name:>8}: excess loss below 0.5 at step {reached}, final {losses[-1]:.3f}")
