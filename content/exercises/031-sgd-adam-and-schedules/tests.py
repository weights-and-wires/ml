"""Checks for the SGD, Adam, and schedules exercise."""

import functools

import numpy as np

from mlcheck import almost, check, close, fail, has


def valley(n=200, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    X = np.sqrt(n) * q * np.array([1.0, 10.0])
    y = X @ np.array([1.0, 0.1]) + rng.normal(size=n)
    return X, y


@check("batch_gradient matches finite differences, and one-row gradients average to the full gradient")
def test_batch_gradient(sub):
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

    X, y = valley(n=40)
    w = np.array([0.3, -0.2])
    singles = np.array([batch_gradient(X, y, w, np.array([i])) for i in range(40)])
    full = X.T @ (X @ w - y) / 40
    close(singles.mean(axis=0), full, atol=1e-9, label="mean of the one-row gradients")


@check("sgd_step subtracts lr times the gradient, keeps the state, and is stable below 2 / 100 on the valley")
def test_sgd_step(sub):
    sgd_step = has(sub, "sgd_step")
    w = np.array([1.0, -2.0, 0.5])
    grad = np.array([0.1, 0.2, -0.3])
    new_w, new_state = sgd_step(w, grad, {"keep": 1}, lr=0.1)
    close(new_w, w - 0.1 * grad, atol=1e-12, label="new weights")
    if new_state != {"keep": 1}:
        fail(f"sgd_step must return the state unchanged, got {new_state}")

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


@check("momentum_step follows v = beta * v + grad, and a steady gradient gives a step of lr / (1 - beta) times it")
def test_momentum_step(sub):
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

    lr, beta = 0.01, 0.9
    w = np.zeros(1)
    state = {}
    for _ in range(300):
        before = w.copy()
        w, state = momentum_step(w, np.array([2.0]), state, lr=lr, beta=beta)
    almost(float(np.abs(w - before)[0]), lr / (1 - beta) * 2.0, tol=1e-6, label="steady step")
    almost(float(state["v"][0]), 20.0, tol=1e-6, label="steady velocity")


@check("ema_update keeps beta of the old average, and bias_correct recovers a constant input and rejects t below 1")
def test_ema_and_bias(sub):
    ema_update = has(sub, "ema_update")
    bias_correct = has(sub, "bias_correct")
    prev = np.array([1.0, -2.0, 4.0])
    x = np.array([3.0, 0.0, 4.0])
    close(ema_update(prev, x, 0.9), np.array([1.2, -1.8, 4.0]), atol=1e-12, label="ema_update at beta 0.9")
    close(ema_update(prev, x, 0.0), x, atol=1e-12, label="beta 0 forgets the past")

    x = np.array([3.0, -1.5])
    for beta in (0.9, 0.999):
        avg = np.zeros(2)
        for t in range(1, 6):
            avg = ema_update(avg, x, beta)
            close(avg, (1 - beta**t) * x, atol=1e-12, label=f"raw average at t={t}")
            close(bias_correct(avg, beta, t), x, atol=1e-9, label=f"corrected average at t={t}")
    try:
        bias_correct(avg, 0.9, 0)
    except ValueError:
        return
    fail("bias_correct with t = 0 should raise ValueError, but it returned a value")


@check("adam_step takes a first step of about the learning rate for gradients 3 and 3000, and matches the formulas")
def test_adam_step(sub):
    adam_step = has(sub, "adam_step")
    for g in (3.0, 3000.0, -3.0):
        w, _ = adam_step(np.zeros(1), np.array([g]), {}, lr=0.1)
        close(w, np.array([-0.1 * np.sign(g)]), atol=1e-6, label=f"first step for gradient {g:g}")
    _, state = adam_step(np.zeros(1), np.array([3.0]), {}, lr=0.1)
    close(state["m"], np.array([0.3]), atol=1e-12, label="raw first moment after one step")
    close(state["v"], np.array([0.009]), atol=1e-12, label="raw second moment after one step")

    rng = np.random.default_rng(0)
    grads = rng.normal(size=(10, 4)) * np.array([1.0, 10.0, 0.1, 100.0])
    b1, b2, eps, lr = 0.9, 0.99, 1e-8, 0.05
    w = rng.normal(size=4)
    w_ref = w.copy()
    m = np.zeros(4)
    v = np.zeros(4)
    state = {}
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


@check("adamw_step differs from adam_step by lr * weight_decay * w, so decay shrinks a quiet and a noisy weight equally")
def test_adamw_step(sub):
    adam_step = has(sub, "adam_step")
    adamw_step = has(sub, "adamw_step")
    rng = np.random.default_rng(1)
    w = rng.normal(size=4)
    grad = rng.normal(size=4)
    lr, wd = 0.05, 0.1
    w_adam, _ = adam_step(w.copy(), grad, {}, lr=lr)
    w_adamw, _ = adamw_step(w.copy(), grad, {}, lr=lr, weight_decay=wd)
    close(w_adam - w_adamw, lr * wd * w, atol=1e-10, label="decoupled decay term")

    wa, wb = w.copy(), w.copy()
    sa: dict = {}
    sb: dict = {}
    for _ in range(4):
        g = rng.normal(size=4)
        wa, sa = adam_step(wa, g, sa, lr=0.1)
        wb, sb = adamw_step(wb, g, sb, lr=0.1, weight_decay=0.0)
        close(wb, wa, atol=1e-12, label="AdamW with zero decay")
    _, s1 = adamw_step(w.copy(), np.array([3.0, 3.0, 3.0, 3.0]), {}, lr=0.1, weight_decay=0.5)
    close(s1["m"], np.full(4, 0.3), atol=1e-12, label="raw first moment, which must not include the decay")

    grads = np.random.default_rng(0).normal(size=(200, 2)) * np.array([0.01, 10.0])
    frozen = np.array([1.0, 1.0])

    def last_step(step_fn, decay):
        state: dict = {}
        for g in grads[:-1]:
            _, state = step_fn(frozen, g, state, lr=0.1, weight_decay=decay)
        return step_fn(frozen, grads[-1], state, lr=0.1, weight_decay=decay)[0]

    shrink = last_step(adamw_step, 0.0) - last_step(adamw_step, 0.01)
    close(shrink, np.array([0.001, 0.001]), atol=1e-9, label="AdamW shrink for the quiet and the noisy weight")

    coupled = lambda w, g, s, lr, weight_decay: adam_step(w, g + weight_decay * w, s, lr)
    quiet, noisy = last_step(coupled, 0.0) - last_step(coupled, 0.01)
    if not quiet > 20 * 0.001:
        fail(f"with L2 inside Adam the quiet weight should shrink by over 20 times 0.001, got {quiet:.5f}")
    if not 0 < noisy < 0.001 / 3:
        fail(f"with L2 inside Adam the noisy weight should shrink by well under 0.001, got {noisy:.6f}")


@check("linear_warmup rises to the peak and holds it, and cosine_decay_fraction is 1, 0.5, and 0 at p = 0, 0.5, and 1")
def test_warmup_and_fraction(sub):
    linear_warmup = has(sub, "linear_warmup")
    f = has(sub, "cosine_decay_fraction")
    almost(linear_warmup(0, 100, 0.01), 0.0, tol=1e-12, label="step 0")
    almost(linear_warmup(50, 100, 0.01), 0.005, tol=1e-12, label="halfway through warmup")
    almost(linear_warmup(100, 100, 0.01), 0.01, tol=1e-12, label="end of warmup")
    almost(linear_warmup(500, 100, 0.01), 0.01, tol=1e-12, label="after warmup")
    almost(linear_warmup(0, 0, 0.01), 0.01, tol=1e-12, label="no warmup at all")
    try:
        linear_warmup(-1, 100, 0.01)
    except ValueError:
        pass
    else:
        fail("linear_warmup with a negative step should raise ValueError")
    almost(f(0.0), 1.0, tol=1e-12, label="p = 0")
    almost(f(0.25), 0.5 * (1 + np.cos(np.pi / 4)), tol=1e-12, label="p = 0.25")
    almost(f(0.5), 0.5, tol=1e-12, label="p = 0.5")
    almost(f(1.0), 0.0, tol=1e-12, label="p = 1")
    almost(f(-0.5), 1.0, tol=1e-12, label="p below 0 clamps to 1")
    almost(f(1.5), 0.0, tol=1e-12, label="p above 1 clamps to 0")


@check("cosine_schedule reproduces the table, never rises during decay, and does not divide by zero when warmup >= total")
def test_cosine_schedule(sub):
    cosine_schedule = has(sub, "cosine_schedule")
    want = {0: 0.0, 50: 0.005, 100: 0.01, 325: 0.0085355339, 550: 0.005, 1000: 0.0}
    for step, rate in want.items():
        almost(cosine_schedule(step, 100, 1000, 0.01), rate, tol=1e-9, label=f"rate at step {step}")
    almost(cosine_schedule(1500, 100, 1000, 0.01), 0.0, tol=1e-12, label="step 1500, past the end")
    rates = np.array([cosine_schedule(s, 100, 1000, 0.01) for s in range(100, 1001)])
    if np.any(np.diff(rates) > 1e-15):
        fail("the rate went up during the decay phase; it should only fall")
    for warmup, total in [(100, 100), (100, 50)]:
        almost(cosine_schedule(warmup // 2, warmup, total, 0.02), 0.01, tol=1e-12, label="halfway through warmup")
        almost(cosine_schedule(warmup + 40, warmup, total, 0.02), 0.02, tol=1e-12, label="after warmup")


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


@check("Adam with warmup and cosine decay ends far below Adam with a constant rate on noisy batches")
def test_schedule_settles(sub):
    run_with_schedule = has(sub, "run_with_schedule")
    cosine_schedule = has(sub, "cosine_schedule")
    adam_step = has(sub, "adam_step")
    X, y = valley()
    best = np.linalg.solve(X.T @ X, X.T @ y)
    excess = lambda w: 0.5 * np.mean((X @ w - y) ** 2) - 0.5 * np.mean((X @ best - y) ** 2)
    steps, peak = 400, 0.1
    schedules = {
        "constant": lambda k: peak,
        "scheduled": functools.partial(cosine_schedule, warmup=20, total=steps, peak=peak),
    }
    tails = {}
    for name, schedule_fn in schedules.items():
        values = []
        for seed in range(4):
            rng = np.random.default_rng(seed)

            def grad_fn(w):
                i = rng.choice(200, 8, replace=False)
                return X[i].T @ (X[i] @ w - y[i]) / 8

            _, history, _ = run_with_schedule(grad_fn, np.array([-1.0, 0.3]), adam_step, steps, schedule_fn)
            values.append(np.mean([excess(w) for w in history[-50:]]))
        tails[name] = float(np.mean(values))
    if not tails["scheduled"] < tails["constant"] / 5:
        fail(f"expected the scheduled run to end at least 5 times lower, got {tails}")


@check("gradient_noise_scale matches hand values, ignores the gradient's units, and is infinite at zero signal")
def test_noise_scale_hand_values(sub):
    gradient_noise_scale = has(sub, "gradient_noise_scale")
    # One weight, gradients 2, -1, 4, 3: G = 2, tr(Sigma) = 14 / 4 = 3.5, so 3.5 / 2**2 = 0.875.
    one = np.array([2.0, -1.0, 4.0, 3.0])
    almost(gradient_noise_scale(one), 0.875, 1e-9, label="noise scale of gradients 2, -1, 4, 3")
    # Two weights: G = (2, 2), Sigma = diag(0.5, 2), so 2.5 / 8 = 0.3125.
    two = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0], [2.0, 2.0]])
    almost(gradient_noise_scale(two), 0.3125, 1e-9, label="noise scale of the four two-weight gradients")
    # Wrong implementation caught here: dividing by |G| instead of |G|^2 gives 3.5 / 2 = 1.75 for the
    # first array and 2.5 / sqrt(8) = 0.884 for the second. Taking the trace of the covariance of the
    # batch mean (tr(Sigma) / n) gives 0.0781 for the second.
    almost(gradient_noise_scale(5.0 * two), 0.3125, 1e-9, label="noise scale after multiplying every gradient by 5")
    got = gradient_noise_scale(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    if got != float("inf"):
        fail(f"expected inf when the mean gradient is zero, got {got}")


@check("The variance of a mini-batch mean, measured over 4000 draws, falls as 1 / B and equals tr(Sigma) / B")
def test_minibatch_variance_measured(sub):
    minibatch_gradient_variance = has(sub, "minibatch_gradient_variance")
    rng = np.random.default_rng(0)
    g = rng.normal(size=(500, 3)) * np.array([1.0, 2.0, 3.0]) + 0.5
    almost(minibatch_gradient_variance(np.array([2.0, -1.0, 4.0, 3.0]), 2), 1.75, 1e-9, label="tr(Sigma) / 2 for gradients 2, -1, 4, 3")
    draw = np.random.default_rng(1)
    measured = {}
    for b in (1, 4, 16, 64):
        means = np.array([g[draw.integers(0, 500, b)].mean(axis=0) for _ in range(4000)])
        measured[b] = float(means.var(axis=0).sum())
        want = measured[b]
        got = minibatch_gradient_variance(g, b)
        if not abs(got - want) < 0.1 * want:
            fail(f"batch size {b}: measured variance of the mean {want:.4f}, but minibatch_gradient_variance returned {got:.4f}")
    ratio = measured[1] / measured[16]
    if not 12.0 < ratio < 20.0:
        fail(f"expected the measured variance to fall 16-fold from B = 1 to B = 16, got a ratio of {ratio:.2f}")
    try:
        minibatch_gradient_variance(g, 0)
    except ValueError:
        return
    fail("expected ValueError for batch_size 0")


@check("On the valley's per-row gradients, the noise scale is a few units at the start and over 50 times larger near the bottom")
def test_noise_scale_valley(sub):
    gradient_noise_scale = has(sub, "gradient_noise_scale")
    X, y = valley()
    best = np.linalg.solve(X.T @ X, X.T @ y)

    def scale_at(w):
        return gradient_noise_scale(X * (X @ w - y)[:, None])

    start = scale_at(np.array([-1.0, 0.3]))
    near = scale_at(best + np.array([0.01, 0.001]))
    if not 2.0 < start < 8.0:
        fail(f"expected a noise scale between 2 and 8 at the start, got {start:.3f}")
    if not near > 50.0 * start:
        fail(f"expected the noise scale near the bottom to exceed 50 times {start:.3f}, got {near:.1f}")
