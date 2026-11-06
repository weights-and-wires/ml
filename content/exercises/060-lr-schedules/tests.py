"""Checks for the learning-rate schedules exercise."""

import functools

import numpy as np

from mlcheck import almost, check, fail, has


@check("linear_warmup rises in a straight line to the peak, then holds it")
def test_linear_warmup(sub):
    linear_warmup = has(sub, "linear_warmup")
    almost(linear_warmup(0, 100, 0.01), 0.0, tol=1e-12, label="step 0")
    almost(linear_warmup(50, 100, 0.01), 0.005, tol=1e-12, label="halfway through warmup")
    almost(linear_warmup(100, 100, 0.01), 0.01, tol=1e-12, label="end of warmup")
    almost(linear_warmup(500, 100, 0.01), 0.01, tol=1e-12, label="after warmup")
    almost(linear_warmup(0, 0, 0.01), 0.01, tol=1e-12, label="no warmup at all")
    try:
        linear_warmup(-1, 100, 0.01)
    except ValueError:
        return
    fail("linear_warmup with a negative step should raise ValueError")


@check("cosine_decay_fraction is 1, 0.8536, 0.5, and 0 at p = 0, 0.25, 0.5, and 1, and clamps outside [0, 1]")
def test_cosine_fraction(sub):
    f = has(sub, "cosine_decay_fraction")
    almost(f(0.0), 1.0, tol=1e-12, label="p = 0")
    almost(f(0.25), 0.5 * (1 + np.cos(np.pi / 4)), tol=1e-12, label="p = 0.25")
    almost(f(0.5), 0.5, tol=1e-12, label="p = 0.5")
    almost(f(1.0), 0.0, tol=1e-12, label="p = 1")
    almost(f(-0.5), 1.0, tol=1e-12, label="p below 0 clamps to 1")
    almost(f(1.5), 0.0, tol=1e-12, label="p above 1 clamps to 0")


@check("cosine_schedule reproduces the chapter's table at steps 0, 50, 100, 325, 550, and 1000")
def test_table(sub):
    cosine_schedule = has(sub, "cosine_schedule")
    want = {0: 0.0, 50: 0.005, 100: 0.01, 325: 0.0085355339, 550: 0.005, 1000: 0.0}
    for step, rate in want.items():
        almost(cosine_schedule(step, 100, 1000, 0.01), rate, tol=1e-9, label=f"rate at step {step}")


@check("Past the last step the rate stays at 0, and it never rises during decay")
def test_shape(sub):
    cosine_schedule = has(sub, "cosine_schedule")
    almost(cosine_schedule(1500, 100, 1000, 0.01), 0.0, tol=1e-12, label="step 1500, past the end")
    rates = np.array([cosine_schedule(s, 100, 1000, 0.01) for s in range(100, 1001)])
    if np.any(np.diff(rates) > 1e-15):
        fail("the rate went up during the decay phase; it should only fall")
    warm = np.array([cosine_schedule(s, 100, 1000, 0.01) for s in range(0, 101)])
    if np.any(np.diff(warm) < 0):
        fail("the rate went down during warmup; it should only rise")


@check("With warmup >= total there is no decay, no division by zero, and the rate holds the peak after warmup")
def test_guard(sub):
    cosine_schedule = has(sub, "cosine_schedule")
    for warmup, total in [(100, 100), (100, 50)]:
        almost(cosine_schedule(0, warmup, total, 0.02), 0.0, tol=1e-12, label=f"step 0, warmup {warmup}, total {total}")
        almost(cosine_schedule(warmup // 2, warmup, total, 0.02), 0.01, tol=1e-12, label="halfway through warmup")
        almost(cosine_schedule(warmup, warmup, total, 0.02), 0.02, tol=1e-12, label="end of warmup")
        almost(cosine_schedule(warmup + 40, warmup, total, 0.02), 0.02, tol=1e-12, label="after warmup")


@check("run_with_schedule feeds the schedule's rate to every step")
def test_lrs_recorded(sub):
    run_with_schedule = has(sub, "run_with_schedule")
    cosine_schedule = has(sub, "cosine_schedule")
    schedule_fn = functools.partial(cosine_schedule, warmup=10, total=50, peak=0.1)
    seen = []

    def step_fn(w, grad, state, lr):
        seen.append(lr)
        return w - lr * grad, state

    _, history, lrs = run_with_schedule(lambda w: w, np.array([1.0]), step_fn, 50, schedule_fn)
    want = [schedule_fn(k) for k in range(50)]
    almost(float(np.max(np.abs(np.array(lrs) - np.array(want)))), 0.0, tol=1e-12, label="recorded rates")
    almost(float(np.max(np.abs(np.array(seen) - np.array(want)))), 0.0, tol=1e-12, label="rates given to step_fn")
    if len(history) != 51:
        fail(f"expected a history of 51 entries, got {len(history)}")


@check("Adam with warmup and cosine decay ends far below Adam with a constant rate on noisy batches")
def test_schedule_settles(sub):
    run_with_schedule = has(sub, "run_with_schedule")
    cosine_schedule = has(sub, "cosine_schedule")
    adam_step = has(sub, "adam_step")
    rng0 = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng0.normal(size=(200, 2)))
    X = np.sqrt(200) * q * np.array([1.0, 10.0])
    y = X @ np.array([1.0, 0.1]) + rng0.normal(size=200)
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
