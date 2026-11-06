"""Checks for the Adam and AdamW exercise."""

import numpy as np

from mlcheck import check, close, fail, has


@check("ema_update keeps beta of the old average, and a zero-started average follows (1 - beta**t) times a constant")
def test_ema(sub):
    ema_update = has(sub, "ema_update")
    prev = np.array([1.0, -2.0, 4.0])
    x = np.array([3.0, 0.0, 4.0])
    close(ema_update(prev, x, 0.9), np.array([1.2, -1.8, 4.0]), atol=1e-12, label="ema_update at beta 0.9")
    close(ema_update(prev, x, 0.0), x, atol=1e-12, label="beta 0 forgets the past")
    close(ema_update(prev, x, 0.5), 0.5 * prev + 0.5 * x, atol=1e-12, label="beta 0.5")
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


@check("adam_step matches a reference built from the formulas over ten random steps, and a zero gradient gives a zero step")
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
    w0 = np.array([1.0, 2.0])
    w, _ = adam_step(w0, np.zeros(2), {}, lr=0.1)
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



@check("adam_l2_step is adam_step run on grad + weight_decay * w")
def test_adam_l2(sub):
    adam_step = has(sub, "adam_step")
    adam_l2_step = has(sub, "adam_l2_step")
    rng = np.random.default_rng(0)
    w = rng.normal(size=4)
    state_a: dict = {}
    state_b: dict = {}
    wa, wb = w.copy(), w.copy()
    for _ in range(5):
        g = rng.normal(size=4)
        wa, state_a = adam_l2_step(wa, g, state_a, lr=0.05, weight_decay=0.1)
        wb, state_b = adam_step(wb, g + 0.1 * wb, state_b, lr=0.05)
        close(wa, wb, atol=1e-12, label="weights")
    close(state_a["m"], state_b["m"], atol=1e-12, label="stored m includes the penalty")


@check("adamw_step differs from adam_step by exactly lr * weight_decay * w, equals it at zero decay, and keeps raw averages")
def test_adamw(sub):
    adam_step = has(sub, "adam_step")
    adamw_step = has(sub, "adamw_step")
    rng = np.random.default_rng(1)
    w = rng.normal(size=4)
    grad = rng.normal(size=4)
    lr, wd = 0.05, 0.1
    w_adam, _ = adam_step(w.copy(), grad, {}, lr=lr)
    w_adamw, _ = adamw_step(w.copy(), grad, {}, lr=lr, weight_decay=wd)
    close(w_adam - w_adamw, lr * wd * w, atol=1e-10, label="decoupled decay term")
    rng = np.random.default_rng(2)
    w = rng.normal(size=3)
    wa, wb = w.copy(), w.copy()
    sa: dict = {}
    sb: dict = {}
    for _ in range(4):
        g = rng.normal(size=3)
        wa, sa = adam_step(wa, g, sa, lr=0.1)
        wb, sb = adamw_step(wb, g, sb, lr=0.1, weight_decay=0.0)
        close(wb, wa, atol=1e-12, label="AdamW with zero decay")
    w1, s1 = adamw_step(w.copy(), np.array([3.0, 3.0, 3.0]), {}, lr=0.1, weight_decay=0.5)
    close(s1["m"], np.array([0.3, 0.3, 0.3]), atol=1e-12, label="raw first moment, which must not include the decay")
    if s1["t"] != 1:
        fail(f"expected state['t'] == 1, got {s1['t']}")


@check("For plain SGD, an L2 penalty in the gradient equals decay applied after the step")
def test_sgd_equivalence(sub):
    sgd_step = has(sub, "sgd_step")
    rng = np.random.default_rng(3)
    w = rng.normal(size=5)
    g = rng.normal(size=5)
    lr, wd = 0.1, 0.01
    coupled, _ = sgd_step(w, g + wd * w, {}, lr=lr)
    plain, _ = sgd_step(w, g, {}, lr=lr)
    close(coupled, plain - lr * wd * w, atol=1e-12, label="L2 versus decoupled decay for SGD")
    w1 = np.array([1.0])
    new_w, _ = sgd_step(w1, np.array([0.51]), {}, lr=0.1)
    close(new_w, np.array([0.949]), atol=1e-12, label="the chapter's worked example")


@check("decay_amount returns the last step without decay minus the last step with decay, with w held fixed")
def test_decay_amount_definition(sub):
    decay_amount = has(sub, "decay_amount")
    seen = []

    def step_fn(w, g, state, lr, weight_decay):
        seen.append(w.copy())
        count = state.get("n", 0) + 1
        return w - lr * g - lr * weight_decay * w * count, {"n": count}

    w = np.array([2.0, -1.0])
    grads = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    got = decay_amount(step_fn, w, grads, lr=0.1, weight_decay=0.5)
    close(got, 0.1 * 0.5 * w * 3, atol=1e-12, label="shrink from decay alone, after three threaded calls")
    for held in seen:
        close(held, w, atol=1e-12, label="the weight passed to step_fn, which must stay fixed")
    if len(seen) != 6:
        fail(f"expected step_fn to be called 3 times per run, 6 in all, got {len(seen)}")


def noisy_grads():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 2)) * np.array([0.01, 10.0])


@check("Decay alone shrinks both weights by lr * weight_decay * w under AdamW, but unevenly under L2 inside Adam")
def test_adamw_uniform_l2_uneven(sub):
    decay_amount = has(sub, "decay_amount")
    adamw_step = has(sub, "adamw_step")
    adam_l2_step = has(sub, "adam_l2_step")
    w = np.array([1.0, 1.0])
    got = decay_amount(adamw_step, w, noisy_grads(), lr=0.1, weight_decay=0.01)
    close(got, np.array([0.001, 0.001]), atol=1e-9, label="AdamW shrink for the quiet and the noisy weight")
    quiet, noisy = decay_amount(adam_l2_step, w, noisy_grads(), lr=0.1, weight_decay=0.01)
    if not quiet > 20 * 0.001:
        fail(f"under L2 the quiet weight should shrink by more than 20 times the intended 0.001, got {quiet:.5f}")
    if not 0 < noisy < 0.001 / 3:
        fail(f"under L2 the noisy weight should shrink by well under the intended 0.001, got {noisy:.6f}")
