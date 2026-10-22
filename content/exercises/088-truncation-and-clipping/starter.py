"""Truncate the backward pass and clip the gradient.

You write the chunk boundaries, a clip by global norm, and the truncated backward
pass that uses them. The forward pass and full backpropagation through time from
earlier chapters are solved for you.
"""

import numpy as np


# Written in "A network that reads one step at a time".
def rnn_cell_forward(
    x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Run one step of the cell.

    x_t is (n, d_x) and h_prev is (n, d_h). W_xh is (d_h, d_x), W_hh is
    (d_h, d_h), and b is (d_h,). Return the new state, shape (n, d_h).
    """
    return np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)


# Written in "A network that reads one step at a time".
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


# Written in "Walk the chain backward: backpropagation through time".
def tanh_backward(dh: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Send a gradient back through tanh.

    h is the output of tanh, h = tanh(z), and dh is the gradient with respect
    to h. Return the gradient with respect to z. Both arrays share one shape.
    """
    return dh * (1.0 - h**2)


# Written in "Walk the chain backward: backpropagation through time".
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


# Written in "Walk the chain backward: backpropagation through time".
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


def chunk_bounds(T: int, k: int) -> list[tuple[int, int]]:
    """Split T steps into chunks of k steps.

    Return a list of (start, end) pairs, with end exclusive, covering 0 .. T in
    order. The last chunk is shorter when k does not divide T. If k is at least
    T, there is one chunk.
    """
    raise NotImplementedError("write this function")


def clip_by_global_norm(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    """Shrink a list of gradient arrays so their combined length is at most max_norm.

    Treat all the arrays together as one long vector, and call its length G.
    If G is at most max_norm, return copies unchanged. Otherwise multiply every
    array by the same factor max_norm / G. Return a new list and leave the
    inputs alone. A gradient that is all zeros must come back as zeros.
    """
    raise NotImplementedError("write this function")


def truncated_bptt(
    dH: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through time in chunks of k steps.

    Run the forward pass once over the whole sequence. Then, for each chunk
    from chunk_bounds, run bptt on that chunk alone, starting from the true
    state that entered it (h0 for the first chunk, the last state of the
    previous chunk after that). Gradient never crosses a chunk boundary.

    Return (dx, dW_xh, dW_hh, db), where the weight gradients add up over chunks.
    """
    raise NotImplementedError("write this function")


if __name__ == "__main__":
    print("chunks for T=6, k=2:", chunk_bounds(6, 2))
    print("chunks for T=7, k=3:", chunk_bounds(7, 3))

    clipped = clip_by_global_norm([np.array([3.0]), np.array([4.0])], max_norm=1.0)
    print("clip [3] and [4] to length 1:", [c.tolist() for c in clipped])

    # The running example, six steps, loss on the last state only.
    W_xh, W_hh, b = np.array([[0.5]]), np.array([[0.8]]), np.zeros(1)
    x = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]).reshape(6, 1, 1)
    h0 = np.zeros((1, 1))
    dH = np.zeros((6, 1, 1))
    dH[-1] = 1.0
    full = bptt(dH, x, h0, W_xh, W_hh, b)
    cut = truncated_bptt(dH, x, h0, W_xh, W_hh, b, k=2)
    print("dx_1, full versus truncated:", round(full[0][0].item(), 4), cut[0][0].item())
    print("dW_hh, full versus truncated:", round(full[2].item(), 4), round(cut[2].item(), 4))
