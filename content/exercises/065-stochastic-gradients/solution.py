"""Reference solution for the noisy-gradients exercise."""

import numpy as np


def make_valley(n: int = 200, seed: int = 0, noise: float = 1.0):
    """Given. Build a regression problem whose loss has curvatures 1 and 100."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    X = np.sqrt(n) * q * np.array([1.0, 10.0])  # X.T @ X / n is diag(1, 100)
    y = X @ np.array([1.0, 0.1]) + noise * rng.normal(size=n)
    return X, y


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
