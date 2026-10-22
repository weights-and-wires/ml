"""Checks for the AdamW exercise."""

import numpy as np

from mlcheck import check, close, fail, has


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


@check("adamw_step differs from adam_step by exactly lr * weight_decay * w")
def test_adamw_gap(sub):
    adam_step = has(sub, "adam_step")
    adamw_step = has(sub, "adamw_step")
    rng = np.random.default_rng(1)
    w = rng.normal(size=4)
    grad = rng.normal(size=4)
    lr, wd = 0.05, 0.1
    w_adam, _ = adam_step(w.copy(), grad, {}, lr=lr)
    w_adamw, _ = adamw_step(w.copy(), grad, {}, lr=lr, weight_decay=wd)
    close(w_adam - w_adamw, lr * wd * w, atol=1e-10, label="decoupled decay term")


@check("adamw_step with weight_decay = 0 is adam_step, and its state holds the raw-gradient averages")
def test_adamw_zero_and_state(sub):
    adam_step = has(sub, "adam_step")
    adamw_step = has(sub, "adamw_step")
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


@check("Under AdamW, decay alone shrinks a quiet weight and a noisy weight by the same lr * weight_decay * w")
def test_adamw_uniform(sub):
    decay_amount = has(sub, "decay_amount")
    adamw_step = has(sub, "adamw_step")
    w = np.array([1.0, 1.0])
    got = decay_amount(adamw_step, w, noisy_grads(), lr=0.1, weight_decay=0.01)
    close(got, np.array([0.001, 0.001]), atol=1e-9, label="AdamW shrink for the quiet and the noisy weight")


@check("Under L2 inside Adam, decay shrinks the quiet weight far too much and the noisy weight far too little")
def test_l2_uneven(sub):
    decay_amount = has(sub, "decay_amount")
    adam_l2_step = has(sub, "adam_l2_step")
    w = np.array([1.0, 1.0])
    quiet, noisy = decay_amount(adam_l2_step, w, noisy_grads(), lr=0.1, weight_decay=0.01)
    if not quiet > 20 * 0.001:
        fail(f"the quiet weight should shrink by more than 20 times the intended 0.001, got {quiet:.5f}")
    if not 0 < noisy < 0.001 / 3:
        fail(f"the noisy weight should shrink by well under the intended 0.001, got {noisy:.6f}")
