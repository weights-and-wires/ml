"""Reference solution: a vanilla RNN, BPTT, and the spectral radius test."""

import numpy as np


def rnn_cell_forward(
    x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    return np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b)


def rnn_forward(
    x: np.ndarray, h0: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b: np.ndarray
) -> np.ndarray:
    t_steps = x.shape[0]
    h_seq = np.empty((t_steps,) + h0.shape, dtype=h0.dtype)
    h_prev = h0
    for t in range(t_steps):
        h_prev = rnn_cell_forward(x[t], h_prev, W_xh, W_hh, b)
        h_seq[t] = h_prev
    return h_seq


def bptt(
    dH: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_steps = x.shape[0]
    h_seq = rnn_forward(x, h0, W_xh, W_hh, b)

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)

    carry = np.zeros_like(h0)  # gradient flowing in from steps after t
    for t in reversed(range(t_steps)):
        dh_t = dH[t] + carry
        dz_t = dh_t * (1.0 - h_seq[t] ** 2)  # tanh'(z_t) = 1 - h_t^2
        h_prev = h_seq[t - 1] if t > 0 else h0

        dW_xh += dz_t.T @ x[t]
        dW_hh += dz_t.T @ h_prev
        db += dz_t.sum(axis=0)
        dx[t] = dz_t @ W_xh
        carry = dz_t @ W_hh

    dh0 = carry
    return dx, dW_xh, dW_hh, db, dh0


def truncated_bptt(
    dH: np.ndarray,
    x: np.ndarray,
    h0: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_steps = x.shape[0]
    h_seq = rnn_forward(x, h0, W_xh, W_hh, b)

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)

    for start in range(0, t_steps, k):
        end = min(start + k, t_steps)
        h0_chunk = h_seq[start - 1] if start > 0 else h0
        dx_chunk, dW_xh_chunk, dW_hh_chunk, db_chunk, _dh0_chunk = bptt(
            dH[start:end], x[start:end], h0_chunk, W_xh, W_hh, b
        )
        dx[start:end] = dx_chunk
        dW_xh += dW_xh_chunk
        dW_hh += dW_hh_chunk
        db += db_chunk

    return dx, dW_xh, dW_hh, db


def clip_by_global_norm(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    global_norm = float(np.sqrt(sum(np.sum(np.asarray(g) ** 2) for g in grads)))
    scale = min(1.0, max_norm / global_norm) if global_norm > 0 else 1.0
    return [g * scale for g in grads]


def gradient_scale_over_time(W: np.ndarray, phi_prime_bound: float, T: int) -> float:
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    return (phi_prime_bound * rho) ** T


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
