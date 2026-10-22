"""Backpropagation through one LSTM step.

The forward step is given, because you wrote it in the chapter "Gates and the
LSTM memory update". You write the sum of the two paths into the cell state, and
then the whole backward step.

Shapes: `x` is (batch, input_size), `h` and `c` are (batch, hidden_size).
"""

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
    """Collect every path by which the loss reaches the new cell state.

    The cell state c_new feeds two places: the next step's cell, which
    delivers `dc_new`, and the hidden output `h_new = o * tanh(c_new)`, which
    delivers a second share through the product and the tanh.

    Args:
        dh_new: Gradient of the loss with respect to h_new, shape (batch, hidden_size).
        dc_new: Gradient of the loss with respect to c_new that arrives along
            the belt from the next step, shape (batch, hidden_size).
        o: Output gate from the forward pass, shape (batch, hidden_size).
        tanh_c_new: tanh of c_new from the forward pass, shape (batch, hidden_size).

    Returns:
        The total gradient with respect to c_new, shape (batch, hidden_size).
    """
    raise NotImplementedError("cell_state_gradient")


def lstm_cell_backward(dh_new: np.ndarray, dc_new: np.ndarray, cache: dict):
    """Backpropagate through one LSTM cell step.

    Args:
        dh_new: Gradient of the loss with respect to h_new, shape (batch, hidden_size).
        dc_new: Gradient of the loss with respect to c_new, shape (batch, hidden_size).
        cache: The cache produced by `lstm_cell_forward`.

    Returns:
        A tuple `(dx, dh_prev, dc_prev, dparams)`.
        dx: Gradient with respect to the input, shape (batch, input_size).
        dh_prev: Gradient with respect to the previous hidden state, shape (batch, hidden_size).
        dc_prev: Gradient with respect to the previous cell state, shape (batch, hidden_size).
        dparams: Dict with the same twelve keys as `params`, each mapped to the
            gradient of that parameter, with the same shape as the parameter.
    """
    raise NotImplementedError("lstm_cell_backward")


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
