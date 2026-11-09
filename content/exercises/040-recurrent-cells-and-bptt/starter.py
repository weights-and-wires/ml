"""A recurrent cell and backpropagation through time.

You write one step of the cell, the loop that reuses it at every step, and a
readout that turns the last state into a loss and a gradient on every state.
Then you write the gradient of one step and the backward loop that chains those
steps together.
"""

import numpy as np



def rnn_cell_forward(
    x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Run one step of the cell.

    x_t is (n, d_x) and h_prev is (n, d_h). W_xh is (d_h, d_x), W_hh is
    (d_h, d_h), and b is (d_h,). Return the new state, shape (n, d_h).
    """
    raise NotImplementedError("write this function")


def rnn_forward(
    x: np.ndarray, h0: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Run the cell over a whole sequence.

    x is (T, n, d_x) and h0 is (n, d_h). Return every state after the first
    step, shape (T, n, d_h): row 0 is h_1, and h_0 is not stored.
    """
    raise NotImplementedError("write this function")


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
    raise NotImplementedError("write this function")


def tanh_backward(dh: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Send a gradient back through tanh.

    h is the output of tanh, h = tanh(z), and dh is the gradient with respect
    to h. Return the gradient with respect to z. Both arrays share one shape.
    """
    raise NotImplementedError("write this function")


def step_backward(
    dh_t: np.ndarray,
    h_t: np.ndarray,
    h_prev: np.ndarray,
    x_t: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through one step of the cell.

    dh_t is the full gradient on h_t, shape (n, d_h): the loss attached to this
    step plus whatever came back from later steps. h_t, h_prev, and x_t are the
    values the forward pass used at this step. Return
    (dx_t, dh_prev, dW_xh, dW_hh, db) with the shapes of x_t, h_prev, W_xh,
    W_hh, and b. The weight gradients are for this one step only.
    """
    raise NotImplementedError("write this function")


def bptt(
    dH: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagation through time.

    dH is (T, n, d_h), the gradient of the loss with respect to each state
    h_1 .. h_T. Return (dx, dW_xh, dW_hh, db, dh0), where dx is (T, n, d_x) and
    dh0 is (n, d_h).
    """
    raise NotImplementedError("write this function")


if __name__ == "__main__":
    # The running example: one number in, one number of state, started from 0.3.
    W_xh = np.array([[0.5]])
    W_hh = np.array([[0.8]])
    b = np.zeros(1)
    x = np.array([1.0, 0.5]).reshape(2, 1, 1)
    h0 = np.full((1, 1), 0.3)

    h_seq = rnn_forward(x, h0, W_xh, W_hh, b)
    print("states h_1, h_2:", np.round(h_seq[:, 0, 0], 4))

    # Read out the last state and score it against a target of 1.
    loss, _ = last_step_loss_and_dH(h_seq, np.array([[2.0]]), np.zeros(1), np.array([[1.0]]))
    print(f"loss: {loss:.6f}")

    dH = np.array([0.0, 1.0]).reshape(2, 1, 1)  # one unit of loss pull on h_2
    dx, dW_xh, dW_hh, db, dh0 = bptt(dH, x, h0, W_xh, W_hh, b)
    print("dW_xh:", dW_xh.item(), "dW_hh:", dW_hh.item(), "db:", db.item())
    print("dx:", dx.ravel(), "dh0:", dh0.item())

    # Nudge W_hh by a tiny amount and see how h_2 moves.
    eps = 1e-6
    up = rnn_forward(x, h0, W_xh, W_hh + eps, b)[-1].item()
    down = rnn_forward(x, h0, W_xh, W_hh - eps, b)[-1].item()
    print("finite difference for W_hh:", (up - down) / (2 * eps))
