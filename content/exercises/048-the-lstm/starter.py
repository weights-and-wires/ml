"""Measure a belt's gradient, then build a gate and one LSTM step.

You build the share of a gradient that a plain net and a belt keep, the gate
that decides how much of a number to let through, a short loop that shows what
the belt does when the gates are held fixed, and the full forward step of an
LSTM cell.

Shapes: `x` is (batch, input_size), `h` and `c` are (batch, hidden_size).
Every weight called "W" has shape (input_size, hidden_size), every "U" has shape
(hidden_size, hidden_size), and every bias "b" has shape (hidden_size,). Write
`x @ W`, with no transpose.
"""

import numpy as np


def gradient_after_steps(factor: float, steps: int) -> np.ndarray:
    """Return the share of the gradient left after each of the first `steps` steps.

    Args:
        factor: The multiplier applied at every step, between 0 and 1.
        steps: How many steps to look back, at least 1.

    Returns:
        An array of shape (steps,). Entry t - 1 holds `factor ** t`, so the
        array starts with `factor` and ends with `factor ** steps`.
    """
    raise NotImplementedError("gradient_after_steps")


def forget_gate_gradient_product(forget_gates: np.ndarray) -> np.ndarray:
    """Return the running product of forget gates along the time axis.

    Args:
        forget_gates: Forget-gate values, shape (T, hidden_size), every entry
            in (0, 1). Row 0 is the gate nearest the loss.

    Returns:
        An array of shape (T, hidden_size). Row t holds
        `forget_gates[0] * forget_gates[1] * ... * forget_gates[t]`, entry by
        entry: the share of the gradient that survives t + 1 steps along the belt.
    """
    raise NotImplementedError("forget_gate_gradient_product")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable logistic sigmoid, elementwise."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def gate(x: np.ndarray, h: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute one gate: a fraction between 0 and 1 for every hidden unit.

    Args:
        x: Input at this step, shape (batch, input_size).
        h: Previous hidden state, shape (batch, hidden_size).
        W: Input weights, shape (input_size, hidden_size).
        U: Hidden weights, shape (hidden_size, hidden_size).
        b: Bias, shape (hidden_size,).

    Returns:
        The sigmoid of `x @ W + h @ U + b`, shape (batch, hidden_size), with
        every entry strictly between 0 and 1.
    """
    raise NotImplementedError("gate")


def carry_cell(c: np.ndarray, steps: int, f: np.ndarray, i: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Apply the belt update `c = f * c + i * g` repeatedly, with fixed gates.

    Args:
        c: Starting cell value, any shape.
        steps: How many times to apply the update, at least 0.
        f: Forget gate, the same shape as `c` or broadcastable to it.
        i: Input gate, broadcastable to the shape of `c`.
        g: Candidate content, broadcastable to the shape of `c`.

    Returns:
        The cell value after `steps` updates, as a float array. With `steps` of
        0 it returns `c` unchanged.
    """
    raise NotImplementedError("carry_cell")


def lstm_cell_forward(x: np.ndarray, h: np.ndarray, c: np.ndarray, params: dict):
    """Run one step of an LSTM cell.

    Args:
        x: Input at this step, shape (batch, input_size).
        h: Previous hidden state, shape (batch, hidden_size).
        c: Previous cell state, shape (batch, hidden_size).
        params: Dict with keys "Wf", "Uf", "bf", "Wi", "Ui", "bi", "Wo", "Uo",
            "bo", "Wg", "Ug", "bg".

    Returns:
        A tuple `(h_new, c_new, cache)`.
        h_new: New hidden state, shape (batch, hidden_size).
        c_new: New cell state, shape (batch, hidden_size).
        cache: A dict holding what the backward pass needs: "x", "h", "c" (the
            inputs to this step), "f", "i", "o", "g" (the three gates and the
            candidate), "c_new", "tanh_c_new", and "params".
    """
    raise NotImplementedError("lstm_cell_forward")


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
