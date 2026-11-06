"""Checks for the exercise on how far a gradient survives."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


@check("gradient_after_steps gives 0.45, 0.2025, 0.0911, 0.041 for four steps")
def test_plain_four_steps(sub):
    fn = has(sub, "gradient_after_steps")
    got = np.asarray(fn(0.45, 4))
    shape(got, (4,), label="gradient_after_steps(0.45, 4)")
    close(got, np.array([0.45, 0.2025, 0.091125, 0.04100625]), atol=1e-9, label="plain net, 4 steps")


@check("gradient_after_steps matches repeated multiplication over 200 steps")
def test_plain_long(sub):
    fn = has(sub, "gradient_after_steps")
    for factor in (0.9, 0.99, 0.995):
        got = np.asarray(fn(factor, 200))
        shape(got, (200,), label=f"gradient_after_steps({factor}, 200)")
        running = 1.0
        want = np.empty(200)
        for t in range(200):
            running *= factor
            want[t] = running
        close(got, want, rtol=1e-9, atol=0.0, label=f"factor {factor}")
    almost_value = np.asarray(fn(0.995, 200))[-1]
    if abs(almost_value - 0.3670) > 1e-3:
        fail(f"0.995 over 200 steps should leave about 0.367, got {almost_value:.4f}")


@check("steps_until_below returns an int, and 6 for factor 0.45 and threshold 0.01")
def test_steps_example(sub):
    fn = has(sub, "steps_until_below")
    got = fn(0.45, 0.01)
    if not isinstance(got, (int, np.integer)):
        fail(f"steps_until_below should return an int, got {type(got).__name__}")
    equal(int(got), 6, label="steps_until_below(0.45, 0.01)")
    equal(int(fn(0.9, 0.01)), 44, label="steps_until_below(0.9, 0.01)")


@check("steps_until_below is strictly below: factor 0.5 and threshold 0.25 needs 3 steps, not 2")
def test_steps_whole_ratio(sub):
    fn = has(sub, "steps_until_below")
    equal(int(fn(0.5, 0.25)), 3, label="steps_until_below(0.5, 0.25): 0.5**2 equals 0.25 and is not below it")
    equal(int(fn(0.5, 0.125)), 4, label="steps_until_below(0.5, 0.125)")


@check("steps_until_below is the first step that crosses: below at t, not below at t - 1")
def test_steps_boundary(sub):
    fn = has(sub, "steps_until_below")
    rng = np.random.default_rng(0)
    for _ in range(40):
        factor = float(rng.uniform(0.3, 0.999))
        threshold = float(10 ** rng.uniform(-8, -0.5))
        t = int(fn(factor, threshold))
        if not factor ** t < threshold:
            fail(f"factor {factor:.4f}, threshold {threshold:.3g}: factor**{t} = "
                 f"{factor ** t:.3g} is not below the threshold")
        if t > 1 and factor ** (t - 1) < threshold:
            fail(f"factor {factor:.4f}, threshold {threshold:.3g}: t = {t} is not the smallest, "
                 f"since factor**{t - 1} = {factor ** (t - 1):.3g} is already below the threshold")


@check("forget_gate_gradient_product turns three gates of 0.9 into 0.9, 0.81, 0.729")
def test_product_three_gates(sub):
    fn = has(sub, "forget_gate_gradient_product")
    got = np.asarray(fn(np.full((3, 1), 0.9)))
    shape(got, (3, 1), label="forget_gate_gradient_product output")
    close(got[:, 0], np.array([0.9, 0.81, 0.729]), atol=1e-12, label="three gates of 0.9")


@check("forget_gate_gradient_product multiplies a different gate at every step, per unit")
def test_product_varying(sub):
    fn = has(sub, "forget_gate_gradient_product")
    rng = np.random.default_rng(1)
    T, m = 12, 4
    gates = rng.uniform(0.6, 0.999, size=(T, m))
    got = np.asarray(fn(gates))
    shape(got, (T, m), label="forget_gate_gradient_product output")
    want = np.empty((T, m))
    for t in range(T):
        want[t] = np.prod(gates[: t + 1], axis=0)
    close(got, want, atol=1e-10, rtol=1e-8, label="running product")


@check("a belt with gates at 0.995 keeps over a third of the gradient at 200 steps; a 0.45 plain net keeps almost nothing")
def test_belt_versus_plain(sub):
    belt_fn = has(sub, "forget_gate_gradient_product")
    plain_fn = has(sub, "gradient_after_steps")
    belt = np.asarray(belt_fn(np.full((200, 3), 0.995)))[-1]
    plain = float(np.asarray(plain_fn(0.45, 200))[-1])
    if belt.min() < 0.3:
        fail(f"a belt of gates at 0.995 should keep about 0.367 after 200 steps, got {belt.min():.3g}")
    if plain > 1e-60:
        fail(f"a plain net with factor 0.45 should keep under 1e-60 after 200 steps, got {plain:.3g}")
    if belt.min() / max(plain, 1e-300) < 1e50:
        fail("the belt should keep at least 1e50 times more gradient than the plain net")
