"""A GRU step, and what each recurrent cell costs.

You write the blend that sits at the heart of a GRU, then the full step with its
reset gate, then a parameter count for three kinds of cell.

Shapes: `x` is (batch, input_size) and `h` is (batch, hidden_size). Every weight
called "W" has shape (input_size, hidden_size), every "U" has shape
(hidden_size, hidden_size), and every bias "b" has shape (hidden_size,).
"""

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
    """Blend the old state with a candidate, using the update gate.

    Args:
        h: Previous hidden state, shape (batch, hidden_size).
        candidate: Proposed new content, shape (batch, hidden_size).
        z: Update gate, entries in (0, 1), shape (batch, hidden_size).

    Returns:
        An array of shape (batch, hidden_size) that keeps the fraction `1 - z`
        of `h` and takes the fraction `z` of `candidate`, entry by entry.
    """
    raise NotImplementedError("gru_update_blend")


def gru_cell_forward(x: np.ndarray, h: np.ndarray, params: dict) -> np.ndarray:
    """Run one step of a GRU cell.

    Args:
        x: Input at this step, shape (batch, input_size).
        h: Previous hidden state, shape (batch, hidden_size).
        params: Dict with keys "Wz", "Uz", "bz" (update gate), "Wr", "Ur", "br"
            (reset gate), and "Wh", "Uh", "bh" (candidate).

    Returns:
        The new hidden state, shape (batch, hidden_size). The reset gate scales
        `h` before the candidate's recurrent matrix sees it.
    """
    raise NotImplementedError("gru_cell_forward")


def parameter_count(kind: str, input_size: int, hidden_size: int) -> int:
    """Count the trainable numbers in one recurrent cell.

    Args:
        kind: "rnn" for a plain recurrent cell, "gru", or "lstm".
        input_size: Size of the input vector, d_x.
        hidden_size: Size of the hidden state, d_h.

    Returns:
        A Python int. One weight set holds `d_x * d_h + d_h * d_h + d_h` numbers.
        A plain RNN has one set, a GRU three, and an LSTM four. Raise
        `ValueError` for any other kind.
    """
    raise NotImplementedError("parameter_count")


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
