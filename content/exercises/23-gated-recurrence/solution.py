"""Reference solution for the LSTM and GRU cell exercise."""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable logistic sigmoid, elementwise."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def lstm_cell_forward(x: np.ndarray, h: np.ndarray, c: np.ndarray, params: dict):
    f = _sigmoid(x @ params["Wf"] + h @ params["Uf"] + params["bf"])
    i = _sigmoid(x @ params["Wi"] + h @ params["Ui"] + params["bi"])
    o = _sigmoid(x @ params["Wo"] + h @ params["Uo"] + params["bo"])
    g = np.tanh(x @ params["Wg"] + h @ params["Ug"] + params["bg"])

    c_new = f * c + i * g
    tanh_c_new = np.tanh(c_new)
    h_new = o * tanh_c_new

    cache = {
        "x": x, "h": h, "c": c,
        "f": f, "i": i, "o": o, "g": g,
        "c_new": c_new, "tanh_c_new": tanh_c_new,
        "params": params,
    }
    return h_new, c_new, cache


def lstm_cell_backward(dh_new: np.ndarray, dc_new: np.ndarray, cache: dict):
    x, h, c = cache["x"], cache["h"], cache["c"]
    f, i, o, g = cache["f"], cache["i"], cache["o"], cache["g"]
    tanh_c_new = cache["tanh_c_new"]
    params = cache["params"]

    # Gradient reaching c_new from downstream, plus what arrives through h_new.
    dc_total = dc_new + dh_new * o * (1.0 - tanh_c_new ** 2)

    do = dh_new * tanh_c_new
    df = dc_total * c
    di = dc_total * g
    dg = dc_total * i
    dc_prev = dc_total * f

    d_pre_f = df * f * (1.0 - f)
    d_pre_i = di * i * (1.0 - i)
    d_pre_o = do * o * (1.0 - o)
    d_pre_g = dg * (1.0 - g ** 2)

    dparams = {
        "Wf": x.T @ d_pre_f, "Uf": h.T @ d_pre_f, "bf": d_pre_f.sum(axis=0),
        "Wi": x.T @ d_pre_i, "Ui": h.T @ d_pre_i, "bi": d_pre_i.sum(axis=0),
        "Wo": x.T @ d_pre_o, "Uo": h.T @ d_pre_o, "bo": d_pre_o.sum(axis=0),
        "Wg": x.T @ d_pre_g, "Ug": h.T @ d_pre_g, "bg": d_pre_g.sum(axis=0),
    }

    dx = (
        d_pre_f @ params["Wf"].T
        + d_pre_i @ params["Wi"].T
        + d_pre_o @ params["Wo"].T
        + d_pre_g @ params["Wg"].T
    )
    dh_prev = (
        d_pre_f @ params["Uf"].T
        + d_pre_i @ params["Ui"].T
        + d_pre_o @ params["Uo"].T
        + d_pre_g @ params["Ug"].T
    )

    return dx, dh_prev, dc_prev, dparams


def gru_cell_forward(x: np.ndarray, h: np.ndarray, params: dict) -> np.ndarray:
    z = _sigmoid(x @ params["Wz"] + h @ params["Uz"] + params["bz"])
    r = _sigmoid(x @ params["Wr"] + h @ params["Ur"] + params["br"])
    candidate = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
    return (1.0 - z) * h + z * candidate


def forget_gate_gradient_product(forget_gates: np.ndarray) -> np.ndarray:
    return np.cumprod(forget_gates, axis=0)


def parameter_count(kind: str, input_size: int, hidden_size: int) -> int:
    per_gate = input_size * hidden_size + hidden_size * hidden_size + hidden_size
    if kind == "lstm":
        return 4 * per_gate
    if kind == "gru":
        return 3 * per_gate
    raise ValueError(f"unknown kind {kind!r}, expected 'lstm' or 'gru'")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch, n, m = 4, 6, 8

    def make_lstm_params(rng):
        keys = {}
        for gate in "figo":
            keys[f"W{gate}"] = 0.3 * rng.normal(size=(n, m))
            keys[f"U{gate}"] = 0.3 * rng.normal(size=(m, m))
            keys[f"b{gate}"] = np.zeros(m)
        return keys

    params = make_lstm_params(rng)
    x = rng.normal(size=(batch, n))
    h = rng.normal(size=(batch, m))
    c = rng.normal(size=(batch, m))

    h_new, c_new, cache = lstm_cell_forward(x, h, c, params)
    print(f"lstm h_new shape {h_new.shape}, c_new shape {c_new.shape}")

    dh_new = rng.normal(size=(batch, m)) * 0.1
    dc_new = rng.normal(size=(batch, m)) * 0.1
    dx, dh_prev, dc_prev, dparams = lstm_cell_backward(dh_new, dc_new, cache)
    print(f"dx shape {dx.shape}, dh_prev shape {dh_prev.shape}")

    forget_gates = np.full((200, m), 0.995)
    product = forget_gate_gradient_product(forget_gates)
    print(f"carousel product after 200 steps, min over units: {product[-1].min():.4f}")

    print(f"lstm parameters: {parameter_count('lstm', n, m)}")
    print(f"gru parameters:  {parameter_count('gru', n, m)}")
