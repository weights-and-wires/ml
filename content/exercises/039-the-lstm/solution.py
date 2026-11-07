"""Reference solution for the belt, gate, and LSTM forward step exercise."""

import numpy as np


def gradient_after_steps(factor: float, steps: int) -> np.ndarray:
    exponents = np.arange(1, steps + 1)
    return float(factor) ** exponents


def forget_gate_gradient_product(forget_gates: np.ndarray) -> np.ndarray:
    return np.cumprod(forget_gates, axis=0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable logistic sigmoid, elementwise."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def gate(x: np.ndarray, h: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _sigmoid(x @ W + h @ U + b)


def carry_cell(c: np.ndarray, steps: int, f: np.ndarray, i: np.ndarray, g: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    for _ in range(steps):
        c = f * c + i * g
    return c


def lstm_cell_forward(x: np.ndarray, h: np.ndarray, c: np.ndarray, params: dict):
    f = gate(x, h, params["Wf"], params["Uf"], params["bf"])
    i = gate(x, h, params["Wi"], params["Ui"], params["bi"])
    o = gate(x, h, params["Wo"], params["Uo"], params["bo"])
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


def _demo_params(rng, n, m):
    params = {}
    for name in "figo":
        params[f"W{name}"] = 0.3 * rng.normal(size=(n, m))
        params[f"U{name}"] = 0.3 * rng.normal(size=(m, m))
        params[f"b{name}"] = np.zeros(m)
    params["bf"] = np.full(m, 2.0)  # open the belt at the start
    return params


if __name__ == "__main__":
    plain = gradient_after_steps(0.45, 200)
    belt = forget_gate_gradient_product(np.full((200, 3), 0.995))
    print(f"plain net after 4 steps:   {plain[:4].round(4)}")
    print(f"plain net after 200 steps: {plain[-1]:.3g}")
    print(f"belt after 200 steps:      {belt[-1].round(4)}")

    rng = np.random.default_rng(0)
    batch, n, m = 4, 6, 8
    params = _demo_params(rng, n, m)
    x = rng.normal(size=(batch, n))
    h = rng.normal(size=(batch, m))
    c = rng.normal(size=(batch, m))

    f = gate(x, h, params["Wf"], params["Uf"], params["bf"])
    print(f"forget gate range: {f.min():.3f} to {f.max():.3f}")

    held = carry_cell(np.array([1.0]), 50, f=np.array([1.0]), i=np.array([0.0]), g=np.array([0.7]))
    print(f"a cell of 1.0 after 50 steps with forget 1, input 0: {held[0]}")

    h_new, c_new, cache = lstm_cell_forward(x, h, c, params)
    print(f"h_new shape {h_new.shape}, c_new shape {c_new.shape}")
