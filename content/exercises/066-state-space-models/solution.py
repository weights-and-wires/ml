"""Reference solution: linear attention, state-space models, and the parallel scan."""

from __future__ import annotations

import numpy as np


def feature_map(x: np.ndarray) -> np.ndarray:
    """Return elu(x) + 1, which is positive for every real x.

    Args:
        x: An array of any shape.

    Returns:
        An array of the same shape: x + 1 where x > 0, and exp(x) elsewhere.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def causal_linear_attention_quadratic(phi_q: np.ndarray, phi_k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Causal linear attention from the T by T table of feature scores.

    Args:
        phi_q: Featurised queries, shape (T, d), all entries positive.
        phi_k: Featurised keys, shape (T, d), all entries positive.
        v: Values, shape (T, dv).

    Returns:
        Outputs of shape (T, dv). Row t is the average of v[0..t], weighted by
        phi_q[t] . phi_k[s] and normalised to sum to 1.
    """
    scores = np.tril(phi_q @ phi_k.T)
    weights = scores / scores.sum(axis=1, keepdims=True)
    return weights @ v


def causal_linear_attention_recurrent(phi_q: np.ndarray, phi_k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The same output as the quadratic form, from a running state.

    Args:
        phi_q: Featurised queries, shape (T, d).
        phi_k: Featurised keys, shape (T, d).
        v: Values, shape (T, dv).

    Returns:
        Outputs of shape (T, dv), computed with a (d, dv) state S and a (d,)
        normaliser z that each step updates before it reads.
    """
    T, d = phi_q.shape
    S = np.zeros((d, v.shape[1]))
    z = np.zeros(d)
    out = np.zeros((T, v.shape[1]))
    for t in range(T):
        S = S + np.outer(phi_k[t], v[t])
        z = z + phi_k[t]
        out[t] = (phi_q[t] @ S) / (phi_q[t] @ z)
    return out


def ssm_kernel(A: np.ndarray, B: np.ndarray, C: np.ndarray, T: int) -> np.ndarray:
    """The impulse response K_j = C A^j B for j = 0 .. T-1.

    Args:
        A: State matrix, shape (n, n).
        B: Input vector, shape (n,).
        C: Output vector, shape (n,).
        T: Number of taps.

    Returns:
        An array of shape (T,).
    """
    K = np.zeros(T)
    v = np.asarray(B, dtype=float).copy()
    for j in range(T):
        K[j] = C @ v
        v = A @ v
    return K


def ssm_recurrent(A: np.ndarray, B: np.ndarray, C: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Run h_t = A h_{t-1} + B x_t, y_t = C h_t from h = 0.

    Args:
        A: (n, n). B: (n,). C: (n,). x: (T,) scalar inputs.

    Returns:
        y of shape (T,).
    """
    h = np.zeros(A.shape[0])
    y = np.zeros(len(x))
    for t in range(len(x)):
        h = A @ h + B * x[t]
        y[t] = C @ h
    return y


def ssm_convolution(A: np.ndarray, B: np.ndarray, C: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The same output as ssm_recurrent, as a causal convolution with the kernel.

    Args:
        A: (n, n). B: (n,). C: (n,). x: (T,) scalar inputs.

    Returns:
        y of shape (T,), with y_t = sum over j <= t of K_j x_{t-j}.
    """
    T = len(x)
    K = ssm_kernel(A, B, C, T)
    return np.convolve(x, K)[:T]


def discretise_zoh(a_diag: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order hold for h' = a h + b x with a diagonal a.

    Args:
        a_diag: Continuous diagonal entries, shape (n,). Entries can be 0.
        b: Continuous input vector, shape (n,).
        dt: Step size, positive.

    Returns:
        (a_bar, b_bar), each of shape (n,): a_bar = exp(dt a) and
        b_bar = (a_bar - 1) / a * b, which is dt * b where a = 0.
    """
    a = np.asarray(a_diag, dtype=float)
    b = np.asarray(b, dtype=float)
    a_bar = np.exp(dt * a)
    small = np.abs(a) < 1e-12
    safe = np.where(small, 1.0, a)
    factor = np.where(small, dt, np.expm1(dt * safe) / safe)
    return a_bar, factor * b


def scan_combine(e1: tuple[np.ndarray, np.ndarray], e2: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compose two steps h -> a h + b, applying e1 first and e2 second.

    Args:
        e1: (a1, b1), arrays of the same shape.
        e2: (a2, b2), arrays of the same shape.

    Returns:
        (a1 * a2, a2 * b1 + b2).
    """
    a1, b1 = e1
    a2, b2 = e2
    return a1 * a2, a2 * b1 + b2


def linear_recurrence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Run h_t = a_t * h_{t-1} + b_t from h = 0, one step at a time.

    Args:
        a: Decay factors, shape (T, n). b: Inputs, shape (T, n).

    Returns:
        h of shape (T, n).
    """
    h = np.zeros(a.shape[1])
    out = np.zeros_like(b, dtype=float)
    for t in range(a.shape[0]):
        h = a[t] * h + b[t]
        out[t] = h
    return out


def parallel_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The same h as linear_recurrence, in ceil(log2 T) rounds (Hillis-Steele).

    Args:
        a: Decay factors, shape (T, n). b: Inputs, shape (T, n).

    Returns:
        h of shape (T, n). Any T works, not only powers of two.
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    T = a.shape[0]
    shift = 1
    while shift < T:
        new_a, new_b = scan_combine((a[:-shift], b[:-shift]), (a[shift:], b[shift:]))
        a = np.concatenate([a[:shift], new_a])
        b = np.concatenate([b[:shift], new_b])
        shift *= 2
    return b


def linear_recurrence_backward(a: np.ndarray, b: np.ndarray, dh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of a loss through h_t = a_t * h_{t-1} + b_t.

    Args:
        a: (T, n). b: (T, n). dh: (T, n), the gradient of the loss with
            respect to each h_t.

    Returns:
        (da, db), each (T, n).
    """
    T = a.shape[0]
    h = linear_recurrence(a, b)
    g = np.zeros_like(dh, dtype=float)
    carry = np.zeros(a.shape[1])
    for t in range(T - 1, -1, -1):
        g[t] = dh[t] + carry
        carry = a[t] * g[t]
    h_prev = np.vstack([np.zeros((1, a.shape[1])), h[:-1]])
    return g * h_prev, g


def selective_ssm(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """A diagonal SSM whose a, b, c change at every step, run with the scan.

    Args:
        x: Scalar inputs, shape (T,).
        a: Per-step decay, shape (T, n).
        b: Per-step input weights, shape (T, n).
        c: Per-step output weights, shape (T, n).

    Returns:
        y of shape (T,), with h_t = a_t * h_{t-1} + b_t * x_t and
        y_t = sum over the n states of c_t * h_t.
    """
    h = parallel_scan(a, b * x[:, None])
    return (c * h).sum(axis=1)


if __name__ == "__main__":
    A = np.array([[0.5]])
    B = np.array([1.0])
    C = np.array([1.0])
    x = np.array([1.0, 2.0, 3.0])
    print("kernel    ", ssm_kernel(A, B, C, 3))
    print("recurrent ", ssm_recurrent(A, B, C, x))
    print("convolved ", ssm_convolution(A, B, C, x))
    a = np.array([[0.5], [2.0], [3.0]])
    b = np.ones((3, 1))
    print("loop scan ", linear_recurrence(a, b).ravel())
    print("tree scan ", parallel_scan(a, b).ravel())
