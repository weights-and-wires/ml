"""Reference solution for the LSTM backward step exercise."""

import numpy as np


# Written in the chapter "Gates and the LSTM memory update", and given here so
# this chapter can start from a working forward pass.
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


def cell_state_gradient(dh_new: np.ndarray, dc_new: np.ndarray, o: np.ndarray, tanh_c_new: np.ndarray) -> np.ndarray:
    return dc_new + dh_new * o * (1.0 - tanh_c_new ** 2)


def lstm_cell_backward(dh_new: np.ndarray, dc_new: np.ndarray, cache: dict):
    x, h, c = cache["x"], cache["h"], cache["c"]
    f, i, o, g = cache["f"], cache["i"], cache["o"], cache["g"]
    tanh_c_new = cache["tanh_c_new"]
    params = cache["params"]

    dc_total = cell_state_gradient(dh_new, dc_new, o, tanh_c_new)

    do = dh_new * tanh_c_new
    df = dc_total * c
    di = dc_total * g
    dg = dc_total * i
    dc_prev = dc_total * f

    da = {
        "f": df * f * (1.0 - f),
        "i": di * i * (1.0 - i),
        "o": do * o * (1.0 - o),
        "g": dg * (1.0 - g ** 2),
    }

    dparams = {}
    dx = np.zeros_like(x, dtype=float)
    dh_prev = np.zeros_like(h, dtype=float)
    for gate_name, d in da.items():
        dparams[f"W{gate_name}"] = x.T @ d
        dparams[f"U{gate_name}"] = h.T @ d
        dparams[f"b{gate_name}"] = d.sum(axis=0)
        dx += d @ params[f"W{gate_name}"].T
        dh_prev += d @ params[f"U{gate_name}"].T

    return dx, dh_prev, dc_prev, dparams


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch, n, m = 4, 6, 8
    params = {}
    for name in "figo":
        params[f"W{name}"] = 0.3 * rng.normal(size=(n, m))
        params[f"U{name}"] = 0.3 * rng.normal(size=(m, m))
        params[f"b{name}"] = np.zeros(m)
    x = rng.normal(size=(batch, n))
    h = rng.normal(size=(batch, m))
    c = rng.normal(size=(batch, m))

    h_new, c_new, cache = lstm_cell_forward(x, h, c, params)
    dh_new = 0.1 * rng.normal(size=(batch, m))
    dc_new = 0.1 * rng.normal(size=(batch, m))

    dc_total = cell_state_gradient(dh_new, dc_new, cache["o"], cache["tanh_c_new"])
    print(f"dc_total shape {dc_total.shape}")

    dx, dh_prev, dc_prev, dparams = lstm_cell_backward(dh_new, dc_new, cache)
    print(f"dx {dx.shape}, dh_prev {dh_prev.shape}, dc_prev {dc_prev.shape}")
    print(f"dparams keys: {sorted(dparams)}")
    print(f"belt check, largest |dc_prev - dc_total * f|: {np.abs(dc_prev - dc_total * cache['f']).max():.2e}")
