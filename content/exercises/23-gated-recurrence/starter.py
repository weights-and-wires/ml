"""The constant error carousel, in code.

You implement one step of an LSTM cell forward and backward, one step of a GRU
cell forward, the cumulative gate product that the lesson derives for the
gradient of the cell state across time, and the parameter count of both
architectures.

Shapes throughout: `x` is `(batch, input_size)`, `h` and `c` are
`(batch, hidden_size)`. A `params` dict holds every weight the cell needs; see
each function's docstring for the exact keys.
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


def lstm_cell_forward(x: np.ndarray, h: np.ndarray, c: np.ndarray, params: dict):
    """Run one step of an LSTM cell.

    Args:
        x: Input at this step, shape (batch, input_size).
        h: Previous hidden state, shape (batch, hidden_size).
        c: Previous cell state, shape (batch, hidden_size).
        params: Dict with keys "Wf", "Uf", "bf", "Wi", "Ui", "bi", "Wo", "Uo",
            "bo", "Wg", "Ug", "bg". Every "W" is shape (input_size,
            hidden_size), every "U" is shape (hidden_size, hidden_size), and
            every "b" is shape (hidden_size,).

    Returns:
        A tuple `(h_new, c_new, cache)`.
        h_new: New hidden state, shape (batch, hidden_size).
        c_new: New cell state, shape (batch, hidden_size).
        cache: A dict holding every quantity `lstm_cell_backward` needs: "x",
            "h", "c" (the inputs to this step), "f", "i", "o", "g" (the four
            gate activations), "c_new", "tanh_c_new", and "params".
    """
    raise NotImplementedError("lstm_cell_forward")


def lstm_cell_backward(dh_new: np.ndarray, dc_new: np.ndarray, cache: dict):
    """Backpropagate through one LSTM cell step.

    Args:
        dh_new: Gradient of the loss with respect to h_new, shape
            (batch, hidden_size).
        dc_new: Gradient of the loss with respect to c_new, shape
            (batch, hidden_size).
        cache: The cache produced by `lstm_cell_forward`.

    Returns:
        A tuple `(dx, dh_prev, dc_prev, dparams)`.
        dx: Gradient with respect to the input, shape (batch, input_size).
        dh_prev: Gradient with respect to the previous hidden state, shape
            (batch, hidden_size).
        dc_prev: Gradient with respect to the previous cell state, shape
            (batch, hidden_size).
        dparams: Dict with the same keys as the `params` argument to
            `lstm_cell_forward`, each mapped to the gradient of that
            parameter, with the same shape as the parameter itself.
    """
    raise NotImplementedError("lstm_cell_backward")


def gru_cell_forward(x: np.ndarray, h: np.ndarray, params: dict) -> np.ndarray:
    """Run one step of a GRU cell.

    Args:
        x: Input at this step, shape (batch, input_size).
        h: Previous hidden state, shape (batch, hidden_size).
        params: Dict with keys "Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh",
            "bh". Every "W" is shape (input_size, hidden_size), every "U" is
            shape (hidden_size, hidden_size), and every "b" is shape
            (hidden_size,).

    Returns:
        The new hidden state, shape (batch, hidden_size).
    """
    raise NotImplementedError("gru_cell_forward")


def forget_gate_gradient_product(forget_gates: np.ndarray) -> np.ndarray:
    """Return the cumulative product of forget-gate values across time.

    This is the quantity the lesson derives for the cell-state Jacobian
    composed across steps: entry `t` of the result is the elementwise product
    of `forget_gates[0], ..., forget_gates[t]`.

    Args:
        forget_gates: Forget-gate values across time, shape (T, hidden_size),
            with every entry in (0, 1).

    Returns:
        An array of the same shape (T, hidden_size), where row t holds
        `forget_gates[0] * forget_gates[1] * ... * forget_gates[t]`,
        elementwise.
    """
    raise NotImplementedError("forget_gate_gradient_product")


def parameter_count(kind: str, input_size: int, hidden_size: int) -> int:
    """Return the number of trainable parameters in one recurrent cell.

    Args:
        kind: Either "lstm" or "gru".
        input_size: Size of the input vector, n.
        hidden_size: Size of the hidden and cell state, m.

    Returns:
        4 * (n*m + m*m + m) for "lstm", 3 * (n*m + m*m + m) for "gru".
    """
    raise NotImplementedError("parameter_count")


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
