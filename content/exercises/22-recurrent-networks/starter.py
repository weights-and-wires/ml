"""A vanilla RNN cell, backpropagation through time, and what decides whether
its gradient vanishes or explodes.

Every sequence is time-major: shape (T, N, ...) with T the number of steps and
N the batch size. The recurrent cell is h_t = tanh(W_xh x_t + W_hh h_{t-1} + b).
"""

import numpy as np


def rnn_cell_forward(
    x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """One step of a vanilla RNN cell.

    Args:
        x_t: Input at this step, shape (N, d_x).
        h_prev: Hidden state from the previous step, shape (N, d_h).
        W_xh: Input-to-hidden weights, shape (d_h, d_x).
        W_hh: Hidden-to-hidden weights, shape (d_h, d_h).
        b: Bias, shape (d_h,).

    Returns:
        The new hidden state, shape (N, d_h).
    """
    raise NotImplementedError("rnn_cell_forward")


def rnn_forward(
    x: np.ndarray, h0: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Run the RNN cell over a full sequence.

    Args:
        x: Input sequence, shape (T, N, d_x).
        h0: Initial hidden state, shape (N, d_h).
        W_xh: Input-to-hidden weights, shape (d_h, d_x).
        W_hh: Hidden-to-hidden weights, shape (d_h, d_h).
        b: Bias, shape (d_h,).

    Returns:
        The hidden state at every step, shape (T, N, d_h). Index 0 is h_1, the
        state after consuming x[0]; h0 itself is not included.
    """
    raise NotImplementedError("rnn_forward")


def bptt(
    dH: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagation through time for the vanilla RNN.

    Args:
        dH: Upstream gradient with respect to every hidden state, shape
            (T, N, d_h): dH[t] is the direct loss gradient attached to h_{t+1}
            (using the same 0-indexing as rnn_forward's output).
        x: The forward pass's input sequence, shape (T, N, d_x).
        h0: The forward pass's initial hidden state, shape (N, d_h).
        W_xh: Input-to-hidden weights, shape (d_h, d_x).
        W_hh: Hidden-to-hidden weights, shape (d_h, d_h).
        b: Bias, shape (d_h,).

    Returns:
        A tuple (dx, dW_xh, dW_hh, db, dh0) with shapes matching
        (x, W_xh, W_hh, b, h0).
    """
    raise NotImplementedError("bptt")


def truncated_bptt(
    dH: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagation through time, truncated to windows of k steps.

    The forward hidden state is the same continuous recursion rnn_forward
    computes. The backward pass runs independently on each consecutive chunk
    of k steps (the last chunk may be shorter), treating the hidden state
    carried in from the previous chunk as a constant: no gradient flows
    across a chunk boundary.

    Args:
        dH: Upstream gradient, shape (T, N, d_h), as in bptt.
        x: Input sequence, shape (T, N, d_x).
        h0: Initial hidden state, shape (N, d_h).
        W_xh: Input-to-hidden weights, shape (d_h, d_x).
        W_hh: Hidden-to-hidden weights, shape (d_h, d_h).
        b: Bias, shape (d_h,).
        k: The truncation window, in steps. k >= the sequence length must
            reduce to exactly the result of bptt.

    Returns:
        A tuple (dx, dW_xh, dW_hh, db) with shapes matching (x, W_xh, W_hh, b).
    """
    raise NotImplementedError("truncated_bptt")


def clip_by_global_norm(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    """Rescale a list of gradients so the norm of their concatenation is capped.

    Args:
        grads: A list of arrays of any shapes.
        max_norm: The largest allowed norm of the concatenation of every array
            in grads, treated as one long vector.

    Returns:
        A new list of arrays, same shapes as grads, each grads[i] multiplied
        by the same scalar min(1, max_norm / global_norm).
    """
    raise NotImplementedError("clip_by_global_norm")


def gradient_scale_over_time(W: np.ndarray, phi_prime_bound: float, T: int) -> float:
    """The Gelfand-formula prediction for a T-step repeated Jacobian product.

    Args:
        W: The recurrent weight matrix, shape (d, d).
        phi_prime_bound: The (assumed constant) activation derivative along
            the path, kappa in the lesson's notation.
        T: Number of steps in the product.

    Returns:
        (phi_prime_bound * rho(W)) ** T, where rho(W) is the spectral radius
        of W: the largest magnitude among its eigenvalues.
    """
    raise NotImplementedError("gradient_scale_over_time")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d_x, d_h, n, t = 3, 5, 4, 6

    W_xh = rng.normal(size=(d_h, d_x)) * 0.3
    W_hh = rng.normal(size=(d_h, d_h)) * 0.3
    b = rng.normal(size=(d_h,)) * 0.1
    h0 = rng.normal(size=(n, d_h)) * 0.1
    x = rng.normal(size=(t, n, d_x)) * 0.5

    h_seq = rnn_forward(x, h0, W_xh, W_hh, b)
    print(f"hidden sequence shape: {h_seq.shape}")

    dH = rng.normal(size=(t, n, d_h)) * 0.1
    dx, dW_xh, dW_hh, db, dh0 = bptt(dH, x, h0, W_xh, W_hh, b)
    print(f"dW_xh shape: {dW_xh.shape}, dW_hh shape: {dW_hh.shape}, dh0 shape: {dh0.shape}")

    clipped = clip_by_global_norm([dW_xh, dW_hh, db], max_norm=1.0)
    global_norm_before = np.sqrt(sum(float(np.sum(g**2)) for g in (dW_xh, dW_hh, db)))
    global_norm_after = np.sqrt(sum(float(np.sum(g**2)) for g in clipped))
    print(f"global norm before/after clipping: {global_norm_before:.3f} / {global_norm_after:.3f}")

    for rho in (0.8, 1.2):
        Q, _ = np.linalg.qr(rng.normal(size=(d_h, d_h)))
        predicted = gradient_scale_over_time(rho * Q, phi_prime_bound=1.0, T=100)
        print(f"rho={rho}: predicted 100-step scale {predicted:.3e}")
