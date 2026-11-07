"""Reference solution: a recurrent cell and backpropagation through time."""

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


def tanh_backward(dh: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Send a gradient back through tanh.

    h is the output of tanh, h = tanh(z), and dh is the gradient with respect
    to h. Return the gradient with respect to z. Both arrays share one shape.
    """
    return dh * (1.0 - h**2)


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
    dz = tanh_backward(dh_t, h_t)
    dx_t = dz @ W_xh
    dh_prev = dz @ W_hh
    dW_xh = dz.T @ x_t
    dW_hh = dz.T @ h_prev
    db = dz.sum(axis=0)
    return dx_t, dh_prev, dW_xh, dW_hh, db


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
    h_seq = rnn_forward(x, h0, W_xh, W_hh, b)
    dx = np.zeros_like(x)
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db = np.zeros_like(b)
    carry = np.zeros_like(h0)
    for t in reversed(range(x.shape[0])):
        h_prev = h_seq[t - 1] if t > 0 else h0
        dx[t], carry, dW_xh_t, dW_hh_t, db_t = step_backward(
            dH[t] + carry, h_seq[t], h_prev, x[t], W_xh, W_hh
        )
        dW_xh += dW_xh_t
        dW_hh += dW_hh_t
        db += db_t
    return dx, dW_xh, dW_hh, db, carry


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
