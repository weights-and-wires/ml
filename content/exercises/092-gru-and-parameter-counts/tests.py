"""Checks for the GRU and parameter count exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gru_case(seed=1, batch=3, n=4, m=5):
    rng = np.random.default_rng(seed)
    params = {}
    for name in "zrh":
        params[f"W{name}"] = 0.4 * rng.normal(size=(n, m))
        params[f"U{name}"] = 0.4 * rng.normal(size=(m, m))
        params[f"b{name}"] = 0.1 * rng.normal(size=m)
    x = 0.5 * rng.normal(size=(batch, n))
    h = 0.5 * rng.normal(size=(batch, m))
    return x, h, params


@check("gru_update_blend returns h at z = 0 and the candidate at z = 1")
def test_blend_ends(sub):
    fn = has(sub, "gru_update_blend")
    rng = np.random.default_rng(0)
    h, cand = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    close(fn(h, cand, np.zeros((3, 4))), h, atol=1e-12, label="blend at z = 0")
    close(fn(h, cand, np.ones((3, 4))), cand, atol=1e-12, label="blend at z = 1")


@check("gru_update_blend gives 0.7546 for h = 0.5, candidate 0.8483, z = 0.7311, and blends per unit")
def test_blend_example(sub):
    fn = has(sub, "gru_update_blend")
    z = _sigmoid(1.0)
    got = fn(np.array([[0.5]]), np.array([[np.tanh(1.25)]]), np.array([[z]]))
    close(got, np.array([[0.7546]]), atol=1e-4, label="worked blend")
    h = np.array([[0.5, 0.5, 0.5]])
    cand = np.array([[0.8, 0.8, 0.8]])
    z3 = np.array([[0.0, 0.5, 1.0]])
    close(fn(h, cand, z3), np.array([[0.5, 0.65, 0.8]]), atol=1e-12, label="per-unit gates")


@check("gru_cell_forward reproduces the worked example: 0.7546")
def test_forward_worked(sub):
    fn = has(sub, "gru_cell_forward")
    one, zero = np.ones((1, 1)), np.zeros((1, 1))
    params = {"Wz": one, "Uz": zero, "bz": np.zeros(1),
              "Wr": zero, "Ur": zero, "br": np.zeros(1),
              "Wh": one, "Uh": one, "bh": np.zeros(1)}
    got = fn(np.array([[1.0]]), np.array([[0.5]]), params)
    shape(got, (1, 1), label="gru_cell_forward output")
    close(got, np.array([[0.7546157427874238]]), atol=1e-9, label="worked GRU step")


@check("gru_cell_forward matches the gate equations on random weights")
def test_forward_equations(sub):
    fn = has(sub, "gru_cell_forward")
    x, h, params = gru_case()
    got = fn(x, h, params)
    shape(got, h.shape, label="gru_cell_forward output")
    z = _sigmoid(x @ params["Wz"] + h @ params["Uz"] + params["bz"])
    r = _sigmoid(x @ params["Wr"] + h @ params["Ur"] + params["br"])
    cand = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
    close(got, (1 - z) * h + z * cand, atol=1e-9, label="gru h_new")


@check("the reset gate acts before the matrix: (r * h) @ Uh, not r * (h @ Uh)")
def test_reset_order(sub):
    fn = has(sub, "gru_cell_forward")
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    big = 40.0  # makes z close to 1, so h_new is close to the candidate
    params = {"Wz": np.zeros((1, 2)), "Uz": np.zeros((2, 2)), "bz": np.full(2, big),
              "Wr": np.zeros((1, 2)), "Ur": np.zeros((2, 2)), "br": np.array([big, -big]),
              "Wh": np.zeros((1, 2)), "Uh": 3.0 * swap, "bh": np.zeros(2)}
    h = np.array([[1.0, 0.0]])
    got = fn(np.zeros((1, 1)), h, params)
    want = np.array([[0.0, np.tanh(3.0)]])
    close(got, want, atol=1e-6, label="GRU step with a swap matrix")


@check("parameter_count gives 10, 30, and 40 when both widths are 2")
def test_count_small(sub):
    fn = has(sub, "parameter_count")
    for kind, want in (("rnn", 10), ("gru", 30), ("lstm", 40)):
        got = fn(kind, 2, 2)
        if not isinstance(got, (int, np.integer)):
            fail(f"parameter_count should return an int, got {type(got).__name__}")
        equal(int(got), want, label=f"parameter_count('{kind}', 2, 2)")


@check("parameter_count at 512 by 512 gives 524,800 / 1,574,400 / 2,099,200, and ratios 1 : 3 : 4")
def test_count_512(sub):
    fn = has(sub, "parameter_count")
    equal(int(fn("rnn", 512, 512)), 524_800, label="rnn")
    equal(int(fn("gru", 512, 512)), 1_574_400, label="gru")
    equal(int(fn("lstm", 512, 512)), 2_099_200, label="lstm")
    for n, m in [(4, 5), (10, 3), (100, 64)]:
        r, g, l = fn("rnn", n, m), fn("gru", n, m), fn("lstm", n, m)
        if g != 3 * r or l != 4 * r:
            fail(f"sizes ({n}, {m}): expected GRU = 3 x RNN and LSTM = 4 x RNN, got {r}, {g}, {l}")


@check("parameter_count rejects an unknown kind with ValueError")
def test_count_unknown(sub):
    fn = has(sub, "parameter_count")
    try:
        fn("transformer", 4, 4)
    except ValueError:
        return
    fail("parameter_count('transformer', 4, 4) should raise ValueError")
