"""Checks for the Adam exercise."""

import numpy as np

from mlcheck import check, close, fail, has


@check("ema_update keeps beta of the old average and 1 - beta of the new value")
def test_ema_update(sub):
    ema_update = has(sub, "ema_update")
    prev = np.array([1.0, -2.0, 4.0])
    x = np.array([3.0, 0.0, 4.0])
    close(ema_update(prev, x, 0.9), np.array([1.2, -1.8, 4.0]), atol=1e-12, label="ema_update at beta 0.9")
    close(ema_update(prev, x, 0.0), x, atol=1e-12, label="beta 0 forgets the past")
    close(ema_update(prev, x, 0.5), 0.5 * prev + 0.5 * x, atol=1e-12, label="beta 0.5")


@check("An average that starts at zero follows (1 - beta**t) times a constant input")
def test_ema_closed_form(sub):
    ema_update = has(sub, "ema_update")
    beta = 0.9
    avg = np.zeros(2)
    x = np.array([3.0, -1.5])
    for t in range(1, 8):
        avg = ema_update(avg, x, beta)
        close(avg, (1 - beta**t) * x, atol=1e-12, label=f"average at t={t}")


@check("bias_correct recovers a constant input exactly and rejects t below 1")
def test_bias_correct(sub):
    ema_update = has(sub, "ema_update")
    bias_correct = has(sub, "bias_correct")
    avg = np.zeros(2)
    x = np.array([3.0, -1.5])
    for t in range(1, 6):
        avg = ema_update(avg, x, 0.999)
        close(bias_correct(avg, 0.999, t), x, atol=1e-9, label=f"corrected average at t={t}")
    try:
        bias_correct(avg, 0.9, 0)
    except ValueError:
        return
    fail("bias_correct with t = 0 should raise ValueError, but it returned a value")


@check("Adam's first step is about the learning rate, whether the gradient is 3 or 3000")
def test_first_step_scale_free(sub):
    adam_step = has(sub, "adam_step")
    for g in (3.0, 3000.0, -3.0):
        w, state = adam_step(np.zeros(1), np.array([g]), {}, lr=0.1)
        want = -0.1 * np.sign(g)
        close(w, np.array([want]), atol=1e-6, label=f"first step for gradient {g:g}")
    _, state = adam_step(np.zeros(1), np.array([3.0]), {}, lr=0.1)
    close(state["m"], np.array([0.3]), atol=1e-12, label="raw first moment after one step")
    close(state["v"], np.array([0.009]), atol=1e-12, label="raw second moment after one step")
    if state["t"] != 1:
        fail(f"expected state['t'] == 1 after one step, got {state['t']}")


@check("adam_step matches a reference built from the formulas over ten random steps")
def test_adam_reference(sub):
    adam_step = has(sub, "adam_step")
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(10, 4)) * np.array([1.0, 10.0, 0.1, 100.0])
    b1, b2, eps, lr = 0.9, 0.99, 1e-8, 0.05
    w = rng.normal(size=4)
    w_ref = w.copy()
    m = np.zeros(4)
    v = np.zeros(4)
    state: dict = {}
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w_ref = w_ref - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        w, state = adam_step(w, g, state, lr=lr, beta1=b1, beta2=b2, eps=eps)
        close(w, w_ref, atol=1e-10, label=f"weights at step {t}")
        close(state["m"], m, atol=1e-10, label=f"stored raw m at step {t}")
        close(state["v"], v, atol=1e-10, label=f"stored raw v at step {t}")
    if state["t"] != 10:
        fail(f"expected state['t'] == 10 after ten steps, got {state['t']}")


@check("A zero gradient gives a zero step and no NaN")
def test_zero_gradient(sub):
    adam_step = has(sub, "adam_step")
    w0 = np.array([1.0, 2.0])
    w, state = adam_step(w0, np.zeros(2), {}, lr=0.1)
    if not np.all(np.isfinite(w)):
        fail(f"a zero gradient produced non-finite weights: {w}")
    close(w, w0, atol=1e-12, label="weights after a zero gradient")


@check("On a valley with curvatures 1 and 10,000, Adam settles in far fewer steps than momentum, and momentum than SGD")
def test_race(sub):
    import functools

    adam_step = has(sub, "adam_step")
    momentum_step = has(sub, "momentum_step")
    sgd_step = has(sub, "sgd_step")
    steps_to_settle = has(sub, "steps_to_settle")
    init = np.array([1.0, 1.0])
    counts = {}
    for steepness in (100.0, 10000.0):
        a = np.array([1.0, steepness])
        grad_fn = lambda w: a * w
        loss_fn = lambda w: 0.5 * float(np.sum(a * w**2))
        for name, step_fn, budget in [
            ("sgd", functools.partial(sgd_step, lr=1.95 / steepness), 30000),
            ("momentum", functools.partial(momentum_step, lr=1.6 / steepness, beta=0.9), 30000),
            ("adam", functools.partial(adam_step, lr=0.05), 3000),
        ]:
            counts[name, steepness] = steps_to_settle(step_fn, grad_fn, loss_fn, init, 1e-4, budget)
    steep = {name: counts[name, 10000.0] for name in ("sgd", "momentum", "adam")}
    if not steep["adam"] < steep["momentum"] < steep["sgd"] <= 30000:
        fail(f"on the steep valley expected Adam < momentum < SGD, got {steep}")
    if counts["adam", 10000.0] > 2 * counts["adam", 100.0]:
        fail(
            "Adam should need about the same number of steps on both valleys, got "
            f"{counts['adam', 100.0]} and {counts['adam', 10000.0]}"
        )
