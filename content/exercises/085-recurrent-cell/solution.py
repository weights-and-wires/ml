"""Build a recurrent cell, run it over a sequence, and give it a loss.

You write one step of the cell, then the loop that reuses it at every step, then a
readout that turns the last state into a loss and a gradient on every state.
"""

import numpy as np


def rnn_cell_forward(
    x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Run one step of the cell.

    x_t is (n, d_x) and h_prev is (n, d_h). W_xh is (d_h, d_x), W_hh is
    (d_h, d_h), and b is (d_h,). Return the new state, shape (n, d_h).
    """
    return np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)


def rnn_forward(
    x: np.ndarray, h0: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Run the cell over a whole sequence.

    x is (T, n, d_x) and h0 is (n, d_h). Return every state after the first
    step, shape (T, n, d_h): row 0 is h_1, and h_0 is not stored.
    """
    h_seq = np.empty((x.shape[0],) + h0.shape, dtype=h0.dtype)
    h_prev = h0
    for t in range(x.shape[0]):
        h_prev = rnn_cell_forward(x[t], h_prev, W_xh, W_hh, b)
        h_seq[t] = h_prev
    return h_seq


def last_step_loss_and_dH(
    h_seq: np.ndarray, W_hy: np.ndarray, b_y: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Read out the last state and score it against the target.

    h_seq is (T, n, d_h), W_hy is (d_o, d_h), b_y is (d_o,), and y is (n, d_o).
    The prediction is y_hat = h_T W_hy^T + b_y, and the loss is
    0.5 * sum((y_hat - y) ** 2) over all examples and outputs.

    Return (loss, dH). dH has the shape of h_seq, holds the gradient of the loss
    with respect to every state, and is zero everywhere except the last step.
    """
    y_hat = h_seq[-1] @ W_hy.T + b_y
    err = y_hat - y
    loss = 0.5 * float(np.sum(err**2))
    dH = np.zeros_like(h_seq)
    dH[-1] = err @ W_hy
    return loss, dH


if __name__ == "__main__":
    # The running example: one number in, one number of state.
    W_xh = np.array([[0.5]])
    W_hh = np.array([[0.8]])
    b = np.zeros(1)
    x = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1, 1)
    h0 = np.zeros((1, 1))

    h_seq = rnn_forward(x, h0, W_xh, W_hh, b)
    print("states h_1 .. h_6:", np.round(h_seq[:, 0, 0], 4))

    # Read out the state after step 2 and score it against a target of 1.
    W_hy = np.array([[2.0]])
    b_y = np.zeros(1)
    loss, dH = last_step_loss_and_dH(h_seq[:2], W_hy, b_y, np.array([[1.0]]))
    print(f"loss after two steps: {loss:.6f}, dH at the last step: {dH[-1, 0, 0]:.4f}")

    d_x, d_h = 8, 16
    print("weights in a cell with d_x=8, d_h=16:", d_h * (d_h + d_x + 1), "for any length T")
