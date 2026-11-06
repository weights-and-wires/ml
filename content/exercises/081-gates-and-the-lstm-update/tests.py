"""Checks for the gate and LSTM forward step exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


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


@check("gate returns shape (n, d_h) with every entry strictly between 0 and 1")
def test_gate_shape_range(sub):
    fn = has(sub, "gate")
    x, h, _, params = lstm_case()
    out = np.asarray(fn(x, h, params["Wf"], params["Uf"], params["bf"]))
    shape(out, h.shape, label="gate output")
    if out.min() <= 0.0 or out.max() >= 1.0:
        fail(f"a gate must stay strictly between 0 and 1, got {out.min():.4g} to {out.max():.4g}")


@check("gate gives 0.8808 for x = 1, h = 0, W = 1, U = 0, b = 1, and 0.5 when everything is 0")
def test_gate_worked_example(sub):
    fn = has(sub, "gate")
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


@check("carry_cell with forget 1 and input 0 keeps the cell exactly; forget 0 replaces it")
def test_carry_hold_and_wipe(sub):
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


@check("carry_cell with forget 0.9 and no input decays the cell by 0.9 per step")
def test_carry_decay(sub):
    fn = has(sub, "carry_cell")
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
