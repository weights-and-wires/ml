"""Checks for the optimisers exercise."""

import functools

import numpy as np

from mlcheck import almost, check, close, fail, has


@check("sgd_step subtracts the learning rate times the gradient")
def test_sgd_step(sub):
    sgd_step = has(sub, "sgd_step")
    w = np.array([1.0, -2.0, 0.5])
    grad = np.array([0.1, 0.2, -0.3])
    new_w, _ = sgd_step(w, grad, {}, lr=0.1)
    close(new_w, w - 0.1 * grad, atol=1e-12, label="sgd_step output")


@check("momentum_step's velocity follows v = beta * v + grad, unnormalised")
def test_momentum_recursion(sub):
    momentum_step = has(sub, "momentum_step")
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(6, 2))
    beta, lr = 0.8, 0.05

    w = np.zeros(2)
    state: dict = {}
    v_manual = np.zeros(2)
    for grad in grads:
        v_manual = beta * v_manual + grad
        want_w = w - lr * v_manual
        w, state = momentum_step(w, grad, state, lr=lr, beta=beta)
        close(w, want_w, atol=1e-10, label="momentum_step trajectory")
        close(state.get("v"), v_manual, atol=1e-10, label="momentum_step velocity state")


@check("momentum's steady-state step size approaches lr / (1 - beta) times the gradient")
def test_momentum_effective_step(sub):
    momentum_step = has(sub, "momentum_step")
    lr, beta = 0.01, 0.9
    g = np.array([2.0])
    w = np.zeros(1)
    state: dict = {}
    w_prev = w.copy()
    for _ in range(3000):
        w_prev = w.copy()
        w, state = momentum_step(w, g, state, lr=lr, beta=beta)
    step_size = float(np.abs(w - w_prev).item())
    want = lr / (1 - beta) * float(g[0])
    almost(step_size, want, tol=1e-3, label="steady-state step size")


@check("adam_step's raw and bias-corrected moments match the closed-form EMA expectation")
def test_adam_bias_correction(sub):
    adam_step = has(sub, "adam_step")
    g = np.array([3.0, -1.5])
    beta1, beta2, lr = 0.9, 0.999, 0.1

    w = np.zeros(2)
    state: dict = {}
    for t in range(1, 6):
        w, state = adam_step(w, g, state, lr=lr, beta1=beta1, beta2=beta2)
        raw_m = state.get("m")
        closed_form_raw = g * (1 - beta1**t)
        close(raw_m, closed_form_raw, atol=1e-10, label=f"raw first moment at t={t}")
        m_hat = raw_m / (1 - beta1**t)
        close(m_hat, g, atol=1e-8, label=f"bias-corrected first moment at t={t}")
        if state.get("t") != t:
            fail(f"expected state['t'] == {t}, got {state.get('t')}")


@check("adamw_step differs from adam_step by exactly the decoupled decay term")
def test_adamw_decoupling(sub):
    adam_step = has(sub, "adam_step")
    adamw_step = has(sub, "adamw_step")
    rng = np.random.default_rng(1)
    w = rng.normal(size=4)
    grad = rng.normal(size=4)
    lr, wd = 0.05, 0.1

    w_adam, _ = adam_step(w.copy(), grad, {}, lr=lr)
    w_adamw, _ = adamw_step(w.copy(), grad, {}, lr=lr, weight_decay=wd)
    close(w_adam - w_adamw, lr * wd * w, atol=1e-10, label="decoupled decay term")

    # With weight_decay = 0, AdamW must reduce to Adam exactly.
    w_adamw_zero, _ = adamw_step(w.copy(), grad, {}, lr=lr, weight_decay=0.0)
    close(w_adamw_zero, w_adam, atol=1e-12, label="AdamW with weight_decay=0")


@check("Adam's update is nearly invariant to rescaling the gradient; SGD's is not")
def test_scale_invariance(sub):
    adam_step = has(sub, "adam_step")
    sgd_step = has(sub, "sgd_step")
    rng = np.random.default_rng(2)
    w = rng.normal(size=3)
    grad = rng.normal(size=3) + 1.0  # keep away from zero
    c = 1000.0

    step_a, _ = adam_step(w.copy(), grad, {}, lr=0.01)
    step_a_scaled, _ = adam_step(w.copy(), c * grad, {}, lr=0.01)
    close(step_a - w, step_a_scaled - w, atol=1e-3, label="Adam's step under a rescaled gradient")

    step_s, _ = sgd_step(w.copy(), grad, {}, lr=0.01)
    step_s_scaled, _ = sgd_step(w.copy(), c * grad, {}, lr=0.01)
    diff = float(np.abs((step_s - w) - (step_s_scaled - w)).max())
    if diff < 1.0:
        fail(
            "sgd_step's update should scale directly with the gradient, but "
            f"rescaling the gradient by {c:g} barely changed it (difference {diff:.3g})"
        )


@check("cosine_schedule hits its endpoints and its peak at the right steps")
def test_cosine_schedule(sub):
    cosine_schedule = has(sub, "cosine_schedule")
    warmup, total, peak = 100, 1000, 0.01
    almost(cosine_schedule(0, warmup, total, peak), 0.0, tol=1e-9, label="step 0")
    almost(cosine_schedule(warmup // 2, warmup, total, peak), peak * 0.5, tol=1e-9, label="mid-warmup")
    almost(cosine_schedule(warmup, warmup, total, peak), peak, tol=1e-9, label="end of warmup")
    almost(cosine_schedule(total, warmup, total, peak), 0.0, tol=1e-9, label="end of schedule")
    midpoint = warmup + (total - warmup) // 2
    almost(cosine_schedule(midpoint, warmup, total, peak), peak * 0.5, tol=1e-6, label="midpoint of decay")


@check("run_optimiser drives sgd_step to the minimum of a well-conditioned quadratic")
def test_run_optimiser_sgd(sub):
    run_optimiser = has(sub, "run_optimiser")
    sgd_step = has(sub, "sgd_step")
    grad_fn = lambda w: w  # loss is 0.5 * ||w||^2
    step_fn = functools.partial(sgd_step, lr=0.1)
    w_final, history = run_optimiser(grad_fn, np.array([3.0, -2.0]), step_fn, steps=200)
    close(w_final, np.zeros(2), atol=1e-3, label="final w")
    if len(history) != 201:
        fail(f"expected a history of length steps + 1 = 201, got {len(history)}")
    close(history[0], np.array([3.0, -2.0]), atol=1e-12, label="history[0] should be the initial point")


@check("SGD, momentum, Adam, and AdamW all reach a low loss on a badly conditioned quadratic")
def test_ill_conditioned(sub):
    run_optimiser = has(sub, "run_optimiser")
    a = np.array([1.0, 100.0])
    grad_fn = lambda w: a * w
    loss_fn = lambda w: 0.5 * float(np.sum(a * w**2))
    init = np.array([1.0, 1.0])
    target = 1e-4

    configs = {
        "sgd": (has(sub, "sgd_step"), {"lr": 0.0095}, 4000),
        "momentum": (has(sub, "momentum_step"), {"lr": 0.004, "beta": 0.9}, 2000),
        "adam": (has(sub, "adam_step"), {"lr": 0.05}, 1500),
        "adamw": (has(sub, "adamw_step"), {"lr": 0.05, "weight_decay": 0.0}, 1500),
    }
    for name, (fn, kwargs, steps) in configs.items():
        step_fn = functools.partial(fn, **kwargs)
        w_final, _ = run_optimiser(grad_fn, init.copy(), step_fn, steps)
        loss = loss_fn(w_final)
        if not np.isfinite(loss) or loss > target:
            fail(
                f"{name} did not reach a loss of {target:g} within {steps} steps on the "
                f"anisotropic quadratic: got {loss:.3g}"
            )
