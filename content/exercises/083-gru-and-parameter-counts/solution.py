"""Reference solution for the GRU and parameter count exercise."""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable logistic sigmoid, elementwise."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def gru_update_blend(h: np.ndarray, candidate: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (1.0 - z) * h + z * candidate


def gru_cell_forward(x: np.ndarray, h: np.ndarray, params: dict) -> np.ndarray:
    z = _sigmoid(x @ params["Wz"] + h @ params["Uz"] + params["bz"])
    r = _sigmoid(x @ params["Wr"] + h @ params["Ur"] + params["br"])
    candidate = np.tanh(x @ params["Wh"] + (r * h) @ params["Uh"] + params["bh"])
    return gru_update_blend(h, candidate, z)


def parameter_count(kind: str, input_size: int, hidden_size: int) -> int:
    per_gate = input_size * hidden_size + hidden_size * hidden_size + hidden_size
    multiplier = {"rnn": 1, "gru": 3, "lstm": 4}
    if kind not in multiplier:
        raise ValueError(f"unknown kind {kind!r}, expected 'rnn', 'gru', or 'lstm'")
    return multiplier[kind] * per_gate


if __name__ == "__main__":
    h = np.array([[0.5]])
    candidate = np.array([[0.8483]])
    print(f"blend at z = 0.7311: {gru_update_blend(h, candidate, np.array([[0.7311]]))[0, 0]:.4f}")

    params = {name: np.zeros((1, 1)) for name in ("Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh", "bh")}
    params["Wz"] = np.ones((1, 1))
    params["Wh"] = np.ones((1, 1))
    params["Uh"] = np.ones((1, 1))
    params["bz"], params["br"], params["bh"] = (np.zeros(1) for _ in range(3))
    h_new = gru_cell_forward(np.array([[1.0]]), h, params)
    print(f"worked-example GRU step: {h_new[0, 0]:.4f}")

    for kind in ("rnn", "gru", "lstm"):
        print(f"{kind:>4} at 512 by 512: {parameter_count(kind, 512, 512):,}")
