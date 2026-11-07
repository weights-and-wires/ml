"""Checks for the belt, gate, and LSTM forward step exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


@check("gradient_after_steps gives 0.45, 0.2025, 0.0911, 0.041 for four steps and matches repeated multiplication over 200")
def test_plain_steps(sub):
    fn = has(sub, "gradient_after_steps")
    got = np.asarray(fn(0.45, 4))
    shape(got, (4,), label="gradient_after_steps(0.45, 4)")
    close(got, np.array([0.45, 0.2025, 0.091125, 0.04100625]), atol=1e-9, label="plain net, 4 steps")
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


@check("forget_gate_gradient_product turns three gates of 0.9 into 0.9, 0.81, 0.729, and multiplies a different gate at every step")
def test_product(sub):
    fn = has(sub, "forget_gate_gradient_product")
    got = np.asarray(fn(np.full((3, 1), 0.9)))
    shape(got, (3, 1), label="forget_gate_gradient_product output")
    close(got[:, 0], np.array([0.9, 0.81, 0.729]), atol=1e-12, label="three gates of 0.9")
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


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_case(seed=0, batch=3, n=4, m=5):
    rng = np.random.default_rng(seed)
    params = {}
    for name in "figo":
        params[f"W{name}"] = 0.4 * rng.normal(size=(n, m))
        params[f"U{name}"] = 0.4 * rng.normal(size=(m, m))
        params[f"b{name}"] = 0.1 * rng.normal(size=m)
    x = 0.5 * rng.normal(size=(batch, n))
    h = 0.5 * rng.normal(size=(batch, m))
    c = 0.5 * rng.normal(size=(batch, m))
    return x, h, c, params


@check("gate returns shape (n, d_h) strictly between 0 and 1, and gives 0.8808 for x = 1, h = 0, W = 1, U = 0, b = 1")
def test_gate_shape_value(sub):
    fn = has(sub, "gate")
    x, h, _, params = lstm_case()
    out = np.asarray(fn(x, h, params["Wf"], params["Uf"], params["bf"]))
    shape(out, h.shape, label="gate output")
    if out.min() <= 0.0 or out.max() >= 1.0:
        fail(f"a gate must stay strictly between 0 and 1, got {out.min():.4g} to {out.max():.4g}")
    one = np.array([[1.0]])
    zero = np.array([[0.0]])
    out = fn(one, zero, one, zero, np.array([1.0]))
    close(out, np.array([[0.8807970779778823]]), atol=1e-9, label="sigma(2)")
    out0 = fn(one, zero, zero, zero, np.array([0.0]))
    close(out0, np.array([[0.5]]), atol=1e-12, label="sigma(0)")


@check("gate matches sigmoid(x @ W + h @ U + b), with no transpose, even when the two widths are equal")
def test_gate_formula(sub):
    fn = has(sub, "gate")
    x, h, _, params = lstm_case(seed=2)
    for name in "fio":
        got = fn(x, h, params[f"W{name}"], params[f"U{name}"], params[f"b{name}"])
        want = _sigmoid(x @ params[f"W{name}"] + h @ params[f"U{name}"] + params[f"b{name}"])
        close(got, want, atol=1e-10, label=f"gate {name}")
    x, h, params = lstm_case(seed=8, n=4, m=4)[0], lstm_case(seed=8, n=4, m=4)[1], lstm_case(seed=8, n=4, m=4)[3]
    got = fn(x, h, params["Wf"], params["Uf"], params["bf"])
    want = _sigmoid(x @ params["Wf"] + h @ params["Uf"] + params["bf"])
    close(got, want, atol=1e-10, label="gate with square W and U (a transposed W runs but is wrong)")


@check("carry_cell holds the cell with forget 1 and input 0, replaces it with forget 0, and decays it by 0.9 per step")
def test_carry(sub):
    fn = has(sub, "carry_cell")
    c = np.array([0.8, -0.3, 2.0])
    ones, zeros = np.ones(3), np.zeros(3)
    g = np.array([0.5, 0.5, 0.5])
    held = np.asarray(fn(c, 100, ones, zeros, g))
    close(held, c, atol=0.0, rtol=0.0, label="cell after 100 held steps")
    wiped = np.asarray(fn(c, 1, zeros, ones, g))
    close(wiped, g, atol=1e-12, label="cell after forget 0, input 1")
    same = np.asarray(fn(c, 0, ones * 0.5, ones, g))
    close(same, c, atol=0.0, rtol=0.0, label="cell after 0 steps")
    c = np.array([1.0, -2.0])
    got = np.asarray(fn(c, 5, np.full(2, 0.9), np.zeros(2), np.zeros(2)))
    close(got, c * 0.9 ** 5, atol=1e-12, label="decayed cell")
    rng = np.random.default_rng(3)
    f, i, g = rng.uniform(0.2, 0.9, 4), rng.uniform(0.0, 1.0, 4), rng.uniform(-1, 1, 4)
    c0 = rng.normal(size=4)
    want = c0.copy()
    for _ in range(7):
        want = f * want + i * g
    close(np.asarray(fn(c0, 7, f, i, g)), want, atol=1e-12, label="seven mixed updates")


@check("lstm_cell_forward reproduces the worked example: c_new = 0.48, h_new = 0.2231")
def test_forward_worked_example(sub):
    fn = has(sub, "lstm_cell_forward")
    zero = np.zeros((1, 1))
    params = {}
    for name in "figo":
        params[f"W{name}"] = np.zeros((1, 1))
        params[f"U{name}"] = np.zeros((1, 1))
    params["bf"] = np.array([np.log(0.9 / 0.1)])   # sigmoid gives 0.9
    params["bi"] = np.array([np.log(0.2 / 0.8)])   # 0.2
    params["bo"] = np.array([0.0])                 # 0.5
    params["bg"] = np.array([np.arctanh(0.6)])     # 0.6
    h_new, c_new, cache = fn(np.array([[1.0]]), zero, np.array([[0.4]]), params)
    close(c_new, np.array([[0.48]]), atol=1e-9, label="c_new")
    close(h_new, np.array([[0.5 * np.tanh(0.48)]]), atol=1e-9, label="h_new")
    close(h_new, np.array([[0.2231]]), atol=1e-4, label="h_new, rounded")


@check("lstm_cell_forward matches the gate equations on random weights and fills the cache")
def test_forward_equations(sub):
    fn = has(sub, "lstm_cell_forward")
    x, h, c, params = lstm_case(seed=5)
    h_new, c_new, cache = fn(x, h, c, params)
    f = _sigmoid(x @ params["Wf"] + h @ params["Uf"] + params["bf"])
    i = _sigmoid(x @ params["Wi"] + h @ params["Ui"] + params["bi"])
    o = _sigmoid(x @ params["Wo"] + h @ params["Uo"] + params["bo"])
    g = np.tanh(x @ params["Wg"] + h @ params["Ug"] + params["bg"])
    want_c = f * c + i * g
    close(c_new, want_c, atol=1e-9, label="c_new")
    close(h_new, o * np.tanh(want_c), atol=1e-9, label="h_new")
    for key in ("x", "h", "c", "f", "i", "o", "g", "c_new", "tanh_c_new", "params"):
        if key not in cache:
            fail(f"the cache is missing '{key}'; the backward pass will need it")
    close(cache["tanh_c_new"], np.tanh(want_c), atol=1e-9, label="cache tanh_c_new")


@check("a forget bias of +20 and an input bias of -20 leave the cell almost untouched")
def test_forward_open_belt(sub):
    fn = has(sub, "lstm_cell_forward")
    x, h, c, params = lstm_case(seed=6)
    params = dict(params)
    params["bf"] = np.full(5, 20.0)
    params["bi"] = np.full(5, -20.0)
    _, c_new, _ = fn(x, h, c, params)
    close(c_new, c, atol=1e-6, label="c_new with the belt held open")
