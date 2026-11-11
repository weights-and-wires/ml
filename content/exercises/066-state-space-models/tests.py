"""Checks for the state-space models and linear attention exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


def _stable_system(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A = 0.9 * A / max(abs(np.linalg.eigvals(A)))  # spectral radius 0.9
    return A, rng.normal(size=n), rng.normal(size=n)


def _loop(a, b):
    h = np.zeros(a.shape[1])
    out = np.zeros_like(b)
    for t in range(a.shape[0]):
        h = a[t] * h + b[t]
        out[t] = h
    return out


# Hand example from the lesson: phi(Q) = [[1,0],[1,1],[0,1]], phi(K) = [[1,0],[0,1],[1,1]], V = [2,4,6].
PQ = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
PK = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
PV = np.array([[2.0], [4.0], [6.0]])


@check("feature_map is elu(x) + 1: positive everywhere and finite at -800 and 800")
def test_feature_map(sub):
    fn = has(sub, "feature_map")
    got = fn(np.array([-800.0, -1.0, 0.0, 2.0, 800.0]))
    close(got, np.array([0.0, np.exp(-1.0), 1.0, 3.0, 801.0]), atol=1e-12, label="feature_map")
    if np.any(got < 0):
        fail(f"feature_map must not be negative, got {got}")


@check("quadratic form on the worked example gives [2, 3, 5], and the future never leaks into the past")
def test_quadratic(sub):
    fn = has(sub, "causal_linear_attention_quadratic")
    got = fn(PQ, PK, PV)
    shape(got, (3, 1), label="output")
    close(got.ravel(), np.array([2.0, 3.0, 5.0]), atol=1e-12, label="worked example")
    rng = np.random.default_rng(0)
    pq, pk, v = np.exp(rng.normal(size=(6, 3))), np.exp(rng.normal(size=(6, 3))), rng.normal(size=(6, 2))
    v2 = v.copy()
    v2[4:] += 10.0
    close(fn(pq, pk, v)[:4], fn(pq, pk, v2)[:4], atol=1e-12, label="rows 0-3 after changing v[4:]")


# Wrong implementation caught here: a recurrence that reads the state before it adds the current key
# and value (off by one). Row 0 then divides 0 by 0, and every later row misses its own step.
@check("recurrent form gives [2, 3, 5] on the worked example and equals the quadratic form to 1e-10")
def test_recurrent_matches(sub):
    phi = has(sub, "feature_map")
    quad = has(sub, "causal_linear_attention_quadratic")
    rec = has(sub, "causal_linear_attention_recurrent")
    got = rec(PQ, PK, PV)
    shape(got, (3, 1), label="output")
    if not np.all(np.isfinite(got)):
        fail(f"output has non-finite values: {got.ravel()} (expected [2, 3, 5])")
    close(got.ravel(), np.array([2.0, 3.0, 5.0]), atol=1e-12, label="worked example")
    rng = np.random.default_rng(1)
    for T, d, dv in [(1, 3, 2), (12, 4, 3), (40, 8, 5)]:
        pq, pk = phi(rng.normal(size=(T, d))), phi(rng.normal(size=(T, d)))
        v = rng.normal(size=(T, dv))
        close(rec(pq, pk, v), quad(pq, pk, v), atol=1e-10, rtol=0, label=f"T={T}")


@check("ssm_kernel gives K = [1, 0.5, 0.25] for a = 0.5, and C A^j B for a random system")
def test_kernel(sub):
    fn = has(sub, "ssm_kernel")
    one = np.array([1.0])
    close(fn(np.array([[0.5]]), one, one, 3), np.array([1.0, 0.5, 0.25]), atol=1e-12, label="scalar kernel")
    A, B, C = _stable_system(3, 2)
    want = np.array([C @ np.linalg.matrix_power(A, j) @ B for j in range(7)])
    close(fn(A, B, C, 7), want, atol=1e-12, label="kernel of a 3-state system")


@check("ssm_recurrent gives [1, 2.5, 4.25] and ssm_convolution matches it to 1e-10")
def test_recurrence_convolution(sub):
    rec = has(sub, "ssm_recurrent")
    conv = has(sub, "ssm_convolution")
    one = np.array([1.0])
    x = np.array([1.0, 2.0, 3.0])
    close(rec(np.array([[0.5]]), one, one, x), np.array([1.0, 2.5, 4.25]), atol=1e-12, label="recurrent, worked")
    close(conv(np.array([[0.5]]), one, one, x), np.array([1.0, 2.5, 4.25]), atol=1e-12, label="convolution, worked")
    A, B, C = _stable_system(4, 3)
    xs = np.random.default_rng(4).normal(size=50)
    close(conv(A, B, C, xs), rec(A, B, C, xs), atol=1e-10, rtol=0, label="convolution vs recurrence, T=50")


@check("discretise_zoh gives (0.3679, 0.6321) for a = -1, and matches a fine simulation of the ODE")
def test_zoh(sub):
    fn = has(sub, "discretise_zoh")
    a_bar, b_bar = fn(np.array([-1.0]), np.array([1.0]), 1.0)
    close(a_bar, np.array([np.exp(-1.0)]), atol=1e-12, label="a_bar")
    close(b_bar, np.array([1.0 - np.exp(-1.0)]), atol=1e-12, label="b_bar")
    a = np.array([-0.5, -2.0, 0.0])
    b = np.array([1.5, -1.0, 2.0])
    dt = 0.7
    a_bar, b_bar = fn(a, b, dt)
    h = np.zeros(3)  # integrate h' = a h + b with x = 1 over dt with tiny Euler steps
    n = 200000
    for _ in range(n):
        h = h + (dt / n) * (a * h + b)
    close(b_bar, h, atol=1e-4, label="b_bar vs simulated h(dt) from h(0)=0")
    close(a_bar, np.exp(dt * a), atol=1e-12, label="a_bar")
    if np.any(np.abs(a_bar[:2]) >= 1):
        fail(f"a negative a must give |a_bar| < 1, got {a_bar}")


# Wrong implementation caught here: composing as (a1 * a2, b1 + b2), which drops the factor a2 on b1.
@check("scan_combine composes (0.5, 1) then (2, 1) into (1, 3), and is associative")
def test_combine(sub):
    fn = has(sub, "scan_combine")
    a, b = fn((np.array([0.5]), np.array([1.0])), (np.array([2.0]), np.array([1.0])))
    close(a, np.array([1.0]), atol=1e-12, label="composed a")
    close(b, np.array([3.0]), atol=1e-12, label="composed b (the wrong b1 + b2 gives 2)")
    rng = np.random.default_rng(5)
    e = [(rng.normal(size=4), rng.normal(size=4)) for _ in range(3)]
    left = fn(fn(e[0], e[1]), e[2])
    right = fn(e[0], fn(e[1], e[2]))
    close(left[0], right[0], atol=1e-12, label="a, (e0 e1) e2 vs e0 (e1 e2)")
    close(left[1], right[1], atol=1e-12, label="b, (e0 e1) e2 vs e0 (e1 e2)")


@check("linear_recurrence and parallel_scan agree with the loop, for T = 1, 8, and 13")
def test_scan(sub):
    seq = has(sub, "linear_recurrence")
    par = has(sub, "parallel_scan")
    close(seq(np.array([[0.5], [2.0], [3.0]]), np.ones((3, 1))).ravel(), np.array([1.0, 3.0, 10.0]), atol=1e-12, label="worked loop")
    close(par(np.array([[0.5], [2.0], [3.0]]), np.ones((3, 1))).ravel(), np.array([1.0, 3.0, 10.0]), atol=1e-12, label="worked scan")
    rng = np.random.default_rng(6)
    for T in (1, 8, 13):
        a = rng.uniform(0.2, 1.1, size=(T, 3))
        b = rng.normal(size=(T, 3))
        want = _loop(a, b)
        close(seq(a, b), want, atol=1e-12, label=f"linear_recurrence T={T}")
        close(par(a, b), want, atol=1e-10, rtol=0, label=f"parallel_scan T={T}")


@check("linear_recurrence_backward matches finite differences for da and db")
def test_backward(sub):
    fn = has(sub, "linear_recurrence_backward")
    rng = np.random.default_rng(7)
    T, n = 6, 2
    a = rng.uniform(0.3, 1.0, size=(T, n))
    b = rng.normal(size=(T, n))
    w = rng.normal(size=(T, n))

    def loss_a(av):
        return float((w * _loop(av.reshape(T, n), b)).sum())

    def loss_b(bv):
        return float((w * _loop(a, bv.reshape(T, n))).sum())

    da, db = fn(a, b, w)
    close(da, finite_difference(loss_a, a.ravel()).reshape(T, n), atol=1e-6, label="da")
    close(db, finite_difference(loss_b, b.ravel()).reshape(T, n), atol=1e-6, label="db")


@check("selective_ssm with input-dependent a, b, c matches a plain loop, and a change at step 3 reaches only later steps")
def test_selective(sub):
    fn = has(sub, "selective_ssm")
    rng = np.random.default_rng(8)
    T, n = 11, 3
    x = rng.normal(size=T)
    a = 1.0 / (1.0 + np.exp(-rng.normal(size=(T, n))))
    b, c = rng.normal(size=(T, n)), rng.normal(size=(T, n))
    h, want = np.zeros(n), np.zeros(T)
    for t in range(T):
        h = a[t] * h + b[t] * x[t]
        want[t] = c[t] @ h
    got = fn(x, a, b, c)
    close(got, want, atol=1e-10, rtol=0, label="selective_ssm")
    # Causality: changing the parameters at step 3 must not touch outputs 0-2.
    a2, b2, c2 = a.copy(), b.copy(), c.copy()
    a2[3] = 0.1
    y2 = fn(x, a2, b2, c2)
    close(y2[:3], got[:3], atol=1e-12, label="outputs 0-2 after changing a[3]")
    if np.allclose(y2[3:], got[3:]):
        fail("changing a[3] left outputs 3 onward unchanged, so the scan ignores a")
