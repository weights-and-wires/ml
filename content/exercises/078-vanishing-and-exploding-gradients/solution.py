"""Predict how far a gradient travels through a recurrent network.

You write the spectral radius, the prediction (kappa * rho) ** T, and the number
of steps a signal survives. The backward pass from the previous chapter is solved
for you, so you can measure a real gradient against your prediction.
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


def spectral_radius(W: np.ndarray) -> float:
    """Return the largest absolute eigenvalue of the square matrix W.

    A real matrix can have complex eigenvalues, so compare their absolute
    values (their distance from zero), not their real parts.
    """
    return float(np.max(np.abs(np.linalg.eigvals(W))))


def gradient_scale_over_time(W: np.ndarray, kappa: float, T: int) -> float:
    """Predict how large a gradient is after flowing back T steps.

    kappa is the typical slope of tanh along the way, a number between 0 and 1.
    Return (kappa * rho) ** T as a float, where rho is the spectral radius of W.
    """
    return float((kappa * spectral_radius(W)) ** T)


def steps_until_below(rho: float, kappa: float, threshold: float) -> int | None:
    """Return the smallest number of steps T with (kappa * rho) ** T < threshold.

    threshold is between 0 and 1. If kappa * rho is 1 or more, the gradient
    never falls below the threshold, so return None.
    """
    rate = kappa * rho
    if rate >= 1.0:
        return None
    if rate <= 0.0:
        return 1
    steps = max(1, int(np.floor(np.log(threshold) / np.log(rate))) + 1)
    while rate**steps >= threshold:  # guard against rounding in the logarithms
        steps += 1
    while steps > 1 and rate ** (steps - 1) < threshold:
        steps -= 1
    return steps


if __name__ == "__main__":
    print("rho of diag(0.5, 0.9):", spectral_radius(np.diag([0.5, 0.9])))
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    print("rho of a quarter turn (no real eigenvalue):", spectral_radius(rotation))
    print("0.9 ** 100 =", gradient_scale_over_time(np.diag([0.5, 0.9]), 1.0, 100))
    print("steps until 0.9 ** T < 0.01:", steps_until_below(0.9, 1.0, 0.01))

    # Measure the gradient that reaches h_0 from a loss on the last of T steps.
    rng = np.random.default_rng(8)
    d, T = 6, 100
    for rho in (0.9, 1.05):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        W_hh = rho * q
        x = np.zeros((T, 1, d))
        dH = np.zeros((T, 1, d))
        dH[-1] = 1.0 / np.sqrt(d)
        h0 = rng.normal(size=(1, d)) * 1e-3
        dh0 = bptt(dH, x, h0, np.zeros((d, d)), W_hh, np.zeros(d))[4]
        print(f"rho={rho}: measured {np.linalg.norm(dh0):.3e}, predicted {rho ** T:.3e}")
