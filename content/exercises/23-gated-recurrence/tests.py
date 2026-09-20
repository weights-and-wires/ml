"""Checks for the LSTM and GRU cell exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_case(seed: int = 0, batch: int = 3, n: int = 4, m: int = 5):
    """A small, well-scaled LSTM cell input and parameter set."""
    rng = np.random.default_rng(seed)
    params = {}
    for gate in "figo":
        params[f"W{gate}"] = 0.4 * rng.normal(size=(n, m))
        params[f"U{gate}"] = 0.4 * rng.normal(size=(m, m))
        params[f"b{gate}"] = 0.1 * rng.normal(size=m)
    x = 0.5 * rng.normal(size=(batch, n))
    h = 0.5 * rng.normal(size=(batch, m))
    c = 0.5 * rng.normal(size=(batch, m))
    return x, h, c, params


def gru_case(seed: int = 1, batch: int = 3, n: int = 4, m: int = 5):
    rng = np.random.default_rng(seed)
    params = {}
    for gate in "zrh":
        params[f"W{gate}"] = 0.4 * rng.normal(size=(n, m))
        params[f"U{gate}"] = 0.4 * rng.normal(size=(m, m))
        params[f"b{gate}"] = 0.1 * rng.normal(size=m)
    x = 0.5 * rng.normal(size=(batch, n))
    h = 0.5 * rng.normal(size=(batch, m))
    return x, h, params


@check("lstm_cell_forward returns the right shapes, with gates in their valid ranges")
def test_lstm_shapes_and_ranges(sub):
    x, h, c, params = lstm_case()
    forward = has(sub, "lstm_cell_forward")
    h_new, c_new, cache = forward(x, h, c, params)
    shape(h_new, h.shape, label="h_new")
    shape(c_new, c.shape, label="c_new")

    for name in ("f", "i", "o"):
        gate = np.asarray(cache[name])
        if gate.min() <= 0.0 or gate.max() >= 1.0:
            fail(f"gate '{name}' from the cache has values outside (0, 1): "
                 f"min {gate.min():.4g}, max {gate.max():.4g}")
    candidate = np.asarray(cache["g"])
    if candidate.min() <= -1.0 or candidate.max() >= 1.0:
        fail(f"candidate 'g' from the cache has values outside (-1, 1): "
             f"min {candidate.min():.4g}, max {candidate.max():.4g}")


@check("lstm_cell_forward matches the hand-derived gate equations")
def test_lstm_forward_matches_equations(sub):
    x, h, c, params = lstm_case(seed=2)
    forward = has(sub, "lstm_cell_forward")
    h_new, c_new, _ = forward(x, h, c, params)

    f = _sigmoid(x @ params["Wf"] + h @ params["Uf"] + params["bf"])
    i = _sigmoid(x @ params["Wi"] + h @ params["Ui"] + params["bi"])
    o = _sigmoid(x @ params["Wo"] + h @ params["Uo"] + params["bo"])
    g = np.tanh(x @ params["Wg"] + h @ params["Ug"] + params["bg"])
    want_c = f * c + i * g
    want_h = o * np.tanh(want_c)

    close(c_new, want_c, atol=1e-8, label="c_new")
    close(h_new, want_h, atol=1e-8, label="h_new")


@check("gru_cell_forward matches the hand-derived gate equations")
def test_gru_forward_matches_equations(sub):
    x, h, params = gru_case()
    forward = has(sub, "gru_cell_forward")
    h_new = forward(x, h, params)
    shape(h_new, h.shape, label="gru h_new")

    z = _sigmoid(x @ params["Wz"] + h @ params["Uz"] + params["bz"])
    r = _sigmoid(x @ params["Wr"] + h @ params["Ur"] + params["br"])
    candidate = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
    want = (1.0 - z) * h + z * candidate
    close(h_new, want, atol=1e-8, label="gru h_new")


@check("lstm_cell_backward matches finite differences for x, h, and c")
def test_lstm_backward_inputs(sub):
    x, h, c, params = lstm_case(seed=3)
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")

    rng = np.random.default_rng(30)
    dh_up = 0.1 * rng.normal(size=h.shape)
    dc_up = 0.1 * rng.normal(size=c.shape)

    _, _, cache = forward(x, h, c, params)
    dx, dh_prev, dc_prev, _ = backward(dh_up, dc_up, cache)

    def loss_of_x(arr):
        h_new, c_new, _ = forward(arr, h, c, params)
        return float(np.sum(h_new * dh_up) + np.sum(c_new * dc_up))

    def loss_of_h(arr):
        h_new, c_new, _ = forward(x, arr, c, params)
        return float(np.sum(h_new * dh_up) + np.sum(c_new * dc_up))

    def loss_of_c(arr):
        h_new, c_new, _ = forward(x, h, arr, params)
        return float(np.sum(h_new * dh_up) + np.sum(c_new * dc_up))

    close(dx, finite_difference(loss_of_x, x), atol=1e-4, rtol=1e-3, label="dx")
    close(dh_prev, finite_difference(loss_of_h, h), atol=1e-4, rtol=1e-3, label="dh_prev")
    close(dc_prev, finite_difference(loss_of_c, c), atol=1e-4, rtol=1e-3, label="dc_prev")


@check("lstm_cell_backward matches finite differences for every weight and bias")
def test_lstm_backward_params(sub):
    x, h, c, params = lstm_case(seed=4)
    forward = has(sub, "lstm_cell_forward")
    backward = has(sub, "lstm_cell_backward")

    rng = np.random.default_rng(40)
    dh_up = 0.1 * rng.normal(size=h.shape)
    dc_up = 0.1 * rng.normal(size=c.shape)

    _, _, cache = forward(x, h, c, params)
    _, _, _, dparams = backward(dh_up, dc_up, cache)

    for name in params:
        def loss_of_param(arr, name=name):
            h_new, c_new, _ = forward(x, h, c, params)
            return float(np.sum(h_new * dh_up) + np.sum(c_new * dc_up))

        grad_fd = finite_difference(loss_of_param, params[name])
        close(dparams[name], grad_fd, atol=1e-4, rtol=1e-3, label=f"gradient wrt {name}")


@check("forget_gate_gradient_product is the elementwise cumulative product across time")
def test_forget_gate_product_correctness(sub):
    fn = has(sub, "forget_gate_gradient_product")
    rng = np.random.default_rng(5)
    T, m = 12, 4
    forget_gates = rng.uniform(0.6, 0.999, size=(T, m))

    got = fn(forget_gates)
    shape(got, (T, m), label="forget_gate_gradient_product output")

    want = np.empty((T, m))
    running = np.ones(m)
    for t in range(T):
        running = running * forget_gates[t]
        want[t] = running
    close(got, want, atol=1e-10, rtol=1e-8, label="cumulative product")


@check("a forget gate pinned near one preserves the gradient over 200 steps; a plain RNN's does not")
def test_carousel_versus_plain_rnn(sub):
    fn = has(sub, "forget_gate_gradient_product")
    T, m = 200, 6

    pinned = np.full((T, m), 0.995)
    carousel = fn(pinned)
    final = carousel[-1]
    if final.min() < 0.1 or final.max() > 1.0:
        fail(
            "with the forget gate pinned at 0.995, the product after 200 steps "
            f"should stay within an order of magnitude of one, but it is "
            f"{final.min():.3g} to {final.max():.3g}"
        )

    # A plain RNN's gradient through T steps is a product of T Jacobians
    # diag(1 - h_t^2) W^T; even a spectral radius safely below one and a
    # generous tanh' of 0.5 compound to numerical zero over the same span.
    plain_rnn_factor = 0.5 * 0.9  # tanh'(h_t) times a sub-unit spectral radius
    plain_rnn_product = plain_rnn_factor ** T
    if plain_rnn_product > 1e-10:
        fail(
            f"a plain RNN factor of {plain_rnn_factor} over {T} steps should "
            f"vanish well past machine tolerance, but the reference value is "
            f"{plain_rnn_product:.3g}"
        )
    if final.min() <= plain_rnn_product:
        fail(
            "the carousel product should stay many orders of magnitude above "
            f"the plain RNN's vanished gradient, but got {final.min():.3g} "
            f"against {plain_rnn_product:.3g}"
        )


@check("parameter_count matches the hand-computed formulas for both architectures")
def test_parameter_count(sub):
    fn = has(sub, "parameter_count")
    for n, m in [(4, 5), (10, 3), (512, 512)]:
        per_gate = n * m + m * m + m
        want_lstm = 4 * per_gate
        want_gru = 3 * per_gate
        got_lstm = fn("lstm", n, m)
        got_gru = fn("gru", n, m)
        if got_lstm != want_lstm:
            fail(f"parameter_count('lstm', {n}, {m}) is {got_lstm}, expected {want_lstm}")
        if got_gru != want_gru:
            fail(f"parameter_count('gru', {n}, {m}) is {got_gru}, expected {want_gru}")
        if abs(got_gru - 0.75 * got_lstm) > 1e-9:
            fail(
                f"the GRU should have exactly 3/4 of the LSTM's parameters for "
                f"the same sizes, but got {got_gru} against {0.75 * got_lstm}"
            )
