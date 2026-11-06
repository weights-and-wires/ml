"""Checks for the noisy-gradients exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has


def valley(n=200, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    X = np.sqrt(n) * q * np.array([1.0, 10.0])
    y = X @ np.array([1.0, 0.1]) + rng.normal(size=n)
    return X, y


@check("batch_gradient matches finite differences of the batch loss")
def test_batch_gradient_finite_difference(sub):
    batch_gradient = has(sub, "batch_gradient")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    w = rng.normal(size=3)
    idx = np.array([2, 5, 7, 11, 20])
    loss = lambda v: 0.5 * np.mean((X[idx] @ v - y[idx]) ** 2)
    num = np.zeros(3)
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-6
        num[j] = (loss(w + e) - loss(w - e)) / 2e-6
    close(batch_gradient(X, y, w, idx), num, atol=1e-6, label="batch_gradient")


@check("The gradients of one-row batches average to the full gradient, so a batch gradient is unbiased")
def test_unbiased(sub):
    batch_gradient = has(sub, "batch_gradient")
    X, y = valley(n=40)
    w = np.array([0.3, -0.2])
    singles = np.array([batch_gradient(X, y, w, np.array([i])) for i in range(40)])
    full = X.T @ (X @ w - y) / 40
    close(singles.mean(axis=0), full, atol=1e-9, label="mean of the one-row gradients")
    all_rows = batch_gradient(X, y, w, np.arange(40))
    close(all_rows, full, atol=1e-9, label="gradient over every row")


@check("sgd_step subtracts lr times the gradient and returns the state unchanged")
def test_sgd_step(sub):
    sgd_step = has(sub, "sgd_step")
    w = np.array([1.0, -2.0, 0.5])
    grad = np.array([0.1, 0.2, -0.3])
    state = {"keep": 1}
    new_w, new_state = sgd_step(w, grad, state, lr=0.1)
    close(new_w, w - 0.1 * grad, atol=1e-12, label="new weights")
    if new_state != {"keep": 1}:
        fail(f"sgd_step must return the state unchanged, got {new_state}")


@check("Plain steps are stable below 2 / 100 = 0.02 on the valley and diverge above it")
def test_stability_limit(sub):
    sgd_step = has(sub, "sgd_step")
    a = np.array([1.0, 100.0])
    for lr, should_converge in [(0.019, True), (0.0205, False)]:
        w = np.array([-1.0, 0.3])
        for _ in range(400):
            w, _ = sgd_step(w, a * w, {}, lr=lr)
        size = float(np.abs(w).max())
        if should_converge and not size < 1.0:
            fail(f"lr {lr} is below 0.02 and should shrink the weights, but they reached size {size:.3g}")
        if not should_converge and not size > 1e3:
            fail(f"lr {lr} is above 0.02 and should diverge, but the weights only reached size {size:.3g}")


@check("momentum_step follows v = beta * v + grad with no (1 - beta) factor")
def test_momentum_recursion(sub):
    momentum_step = has(sub, "momentum_step")
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(6, 2))
    beta, lr = 0.8, 0.05
    w = np.zeros(2)
    state: dict = {}
    v = np.zeros(2)
    for g in grads:
        v = beta * v + g
        want = w - lr * v
        w, state = momentum_step(w, g, state, lr=lr, beta=beta)
        close(w, want, atol=1e-10, label="weights")
        close(state.get("v"), v, atol=1e-10, label="stored velocity")


@check("With a steady gradient, momentum's step approaches lr / (1 - beta) times the gradient")
def test_momentum_steady_state(sub):
    momentum_step = has(sub, "momentum_step")
    lr, beta = 0.01, 0.9
    g = np.array([2.0])
    w = np.zeros(1)
    state: dict = {}
    for _ in range(300):
        before = w.copy()
        w, state = momentum_step(w, g, state, lr=lr, beta=beta)
    almost(float(np.abs(w - before)[0]), lr / (1 - beta) * 2.0, tol=1e-6, label="steady step")
    almost(float(state["v"][0]), 20.0, tol=1e-6, label="steady velocity")


@check("On random batches of eight rows, momentum reaches a low loss in fewer steps than SGD")
def test_race(sub):
    batch_gradient = has(sub, "batch_gradient")
    sgd_step = has(sub, "sgd_step")
    momentum_step = has(sub, "momentum_step")
    X, y = valley()
    n = len(y)
    best = np.linalg.solve(X.T @ X, X.T @ y)
    loss = lambda v: 0.5 * np.mean((X @ v - y) ** 2) - 0.5 * np.mean((X @ best - y) ** 2)

    def steps_to_half(step_fn, seed):
        rng = np.random.default_rng(seed)
        w = np.array([-1.0, 0.3])
        state: dict = {}
        for t in range(1, 400):
            g = batch_gradient(X, y, w, rng.choice(n, 8, replace=False))
            w, state = step_fn(w, g, state)
            if loss(w) < 0.5:
                return t
        return 400

    sgd = np.mean([steps_to_half(lambda w, g, s: sgd_step(w, g, s, lr=0.0095), seed) for seed in range(5)])
    mom = np.mean(
        [steps_to_half(lambda w, g, s: momentum_step(w, g, s, lr=0.0025, beta=0.9), seed) for seed in range(5)]
    )
    if sgd >= 400:
        fail("SGD never reached an excess loss of 0.5; check sgd_step and batch_gradient")
    if not mom < 0.75 * sgd:
        fail(f"expected momentum to need clearly fewer steps than SGD, got {mom:.0f} against {sgd:.0f}")
