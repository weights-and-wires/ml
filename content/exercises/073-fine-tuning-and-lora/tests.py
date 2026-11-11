"""Checks for the fine-tuning and LoRA exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has


def _problem(seed, d_in=7, d_out=6, r=2):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d_out, d_in)) * 0.1
    A = rng.normal(size=(r, d_in)) * 0.3
    B = rng.normal(size=(d_out, r)) * 0.3
    target = rng.normal(size=(d_out, d_in)) * 0.3
    return W, A, B, target


@check("adam_bytes_breakdown lists the five pieces, and they add up to 16 bytes")
def test_breakdown(sub):
    got = has(sub, "adam_bytes_breakdown")()
    want = {
        "weight": 2,
        "gradient": 2,
        "master_weight": 4,
        "first_average": 4,
        "second_average": 4,
    }
    if set(got) != set(want):
        fail(f"expected keys {sorted(want)}, got {sorted(got)}")
    for key, nbytes in want.items():
        if got[key] != nbytes:
            fail(f"{key!r} should be {nbytes} bytes, got {got[key]}")
    equal(sum(got.values()), 16, label="bytes per trained parameter")


@check("trainable_fraction_memory gives 14.98 GB for 1% of 7e9, 112 GB for all, and rejects too many")
def test_fraction(sub):
    fraction = has(sub, "trainable_fraction_memory")
    got = fraction(7_000_000_000, 70_000_000)
    want = 16 * 70_000_000 + 2 * 6_930_000_000
    if got != want:
        fail(f"expected {want:,} bytes (about 14.98 GB) for 1% of 7e9 trained, got {got:,}")
    got = fraction(7_000_000_000, 7_000_000_000)
    if got != 112_000_000_000:
        fail(f"training all 7e9 parameters should cost 112,000,000,000 bytes, got {got:,}")
    rng = np.random.default_rng(1)
    for total in rng.integers(1, 10**9, size=4):
        total = int(total)
        equal(fraction(total, 0), 2 * total, label="train nothing")
    try:
        fraction(100, 101)
    except ValueError:
        return
    fail("trained_params larger than total_params should raise ValueError")


@check("lora_parameter_count returns r * (d_in + d_out)")
def test_parameter_count(sub):
    count = has(sub, "lora_parameter_count")
    for d_in, d_out, r, want in [
        (4096, 4096, 8, 65_536),
        (2, 2, 1, 4),
        (512, 2048, 16, 40_960),
        (128, 128, 1, 256),
    ]:
        got = count(d_in, d_out, r)
        if got != want:
            fail(f"lora_parameter_count({d_in}, {d_out}, {r}) is {got:,}, expected {want:,}")


@check("lora_init returns A of shape (r, d_in) with scale 1/sqrt(d_in), and B of zeros")
def test_init_shapes(sub):
    init = has(sub, "lora_init")
    d_in, d_out, r = 400, 30, 20
    A, B = init(d_in, d_out, r, 0)
    equal(np.shape(A), (r, d_in), label="shape of A")
    equal(np.shape(B), (d_out, r), label="shape of B")
    if np.any(np.asarray(B) != 0):
        fail("B must start as all zeros so the layer begins identical to the frozen one")
    spread = float(np.std(A))
    want = 1 / np.sqrt(d_in)
    if not 0.9 * want < spread < 1.1 * want:
        fail(f"A should have standard deviation about {want:.4f}, got {spread:.4f}")
    if abs(float(np.mean(A))) > 0.1 * want:
        fail(f"A should have mean near 0, got {float(np.mean(A)):.5f}")


@check("lora_forward gives [7, 1] on the 2 by 2 example, and reproduces the frozen layer when B is zero")
def test_forward(sub):
    forward = has(sub, "lora_forward")
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    close(forward(x, W, A, B, 2.0, 1), np.array([[7.0, 1.0]]), label="hand example")
    rng = np.random.default_rng(0)
    d_in, d_out, r = 10, 8, 3
    x = rng.normal(size=(6, d_in))
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = np.zeros((d_out, r))
    close(forward(x, W, A, B, 4.0, r), x @ W.T, rtol=1e-12, atol=1e-12, label="B = 0 output")


@check("merge_lora gives [[3, 4], [0, 1]], has the shape of W, adds a rank-r change, and matches the split path")
def test_merge(sub):
    merge = has(sub, "merge_lora")
    forward = has(sub, "lora_forward")
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    close(merge(W, A, B, 2.0, 1), np.array([[3.0, 4.0], [0.0, 1.0]]), label="merged weight")

    rng = np.random.default_rng(0)
    d_in, d_out, r, alpha = 8, 5, 3, 6.0
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = rng.normal(size=(d_out, r))
    got = merge(W, A, B, alpha, r)
    equal(np.shape(got), (d_out, d_in), label="shape of the merged weight")
    close(got, W + (alpha / r) * (B @ A), rtol=1e-12, atol=1e-12, label="merged weight")
    rank = int(np.linalg.matrix_rank(got - W))
    if rank != r:
        fail(f"merged minus W should have rank {r}, got rank {rank}")

    x = rng.normal(size=(7, d_in))
    close(x @ got.T, forward(x, W, A, B, alpha, r), rtol=1e-10, atol=1e-12,
          label="merged vs split outputs")


@check("best_rank_approximation of [[2, 1], [1, 2]] is all 1.5, and in general has rank r and the Eckart-Young error")
def test_best_rank(sub):
    approx = has(sub, "best_rank_approximation")
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    close(approx(M, 1), np.full((2, 2), 1.5), rtol=1e-9, atol=1e-9, label="rank-1 approximation")

    rng = np.random.default_rng(2)
    m, n, r = 20, 14, 4
    u, _ = np.linalg.qr(rng.normal(size=(m, n)))
    v, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = 10.0 * 0.6 ** np.arange(n)
    delta_w = (u * s) @ v.T
    got = approx(delta_w, r)
    equal(np.shape(got), (m, n), label="shape")
    rank = int(np.linalg.matrix_rank(got, tol=1e-9))
    if rank != r:
        fail(f"the approximation should have rank {r}, got rank {rank}")
    error = float(np.linalg.norm(delta_w - got))
    want = float(np.sqrt(np.sum(s[r:] ** 2)))
    close(error, want, rtol=1e-8, atol=1e-10, label="Frobenius error at rank r")

    rng = np.random.default_rng(3)
    D = rng.normal(size=(12, 9))
    r = 3
    best = float(np.linalg.norm(D - approx(D, r)))
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    worst = (U[:, -r:] * sv[-r:]) @ Vt[-r:]
    if best >= float(np.linalg.norm(D - worst)):
        fail("keeping the smallest singular values should be worse than your approximation")
    for _ in range(20):
        rand = rng.normal(size=(12, r)) @ rng.normal(size=(r, 9))
        if best > float(np.linalg.norm(D - rand)):
            fail("a random rank-r matrix came closer than your approximation")


@check("energy_captured matches the singular values and the approximation error")
def test_energy(sub):
    energy = has(sub, "energy_captured")
    approx = has(sub, "best_rank_approximation")
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    close(energy(M, 1), 0.9, rtol=1e-9, atol=1e-9, label="energy at r = 1")
    close(energy(M, 2), 1.0, rtol=1e-9, atol=1e-9, label="energy at r = 2")
    rng = np.random.default_rng(4)
    D = rng.normal(size=(10, 8))
    last = 0.0
    for r in range(0, 9):
        e = energy(D, r)
        if e < last - 1e-12:
            fail(f"energy should not fall as r grows, but it fell at r = {r}")
        last = e
        if r:
            err = float(np.linalg.norm(D - approx(D, r))) ** 2
            close(e, 1 - err / float(np.linalg.norm(D)) ** 2, rtol=1e-8, atol=1e-10,
                  label=f"energy at r = {r}")


@check("lora_gradients gives dA = [[4, 6]] and dB = [[16], [0]], matches finite differences, and has dA = 0 when B = 0")
def test_gradients(sub):
    grads = has(sub, "lora_gradients")
    merge = has(sub, "merge_lora")
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    target = np.array([[1.0, 1.0], [2.0, 0.0]])
    dA, dB = grads(W, A, B, 2.0, 1, target)
    close(dA, np.array([[4.0, 6.0]]), label="dA")
    close(dB, np.array([[16.0], [0.0]]), label="dB")

    W, A, B, target = _problem(5)
    alpha, r = 4.0, 2

    def loss(a, b):
        return 0.5 * float(np.sum((merge(W, a, b, alpha, r) - target) ** 2))

    dA, dB = grads(W, A, B, alpha, r, target)
    equal(np.shape(dA), np.shape(A), label="shape of dA")
    equal(np.shape(dB), np.shape(B), label="shape of dB")
    num_dA = finite_difference(lambda a: loss(a, B), A.copy(), eps=1e-5)
    num_dB = finite_difference(lambda b: loss(A, b), B.copy(), eps=1e-5)
    close(dA, num_dA, rtol=1e-4, atol=1e-7, label="dA vs finite differences")
    close(dB, num_dB, rtol=1e-4, atol=1e-7, label="dB vs finite differences")

    W, A, _, target = _problem(6)
    B = np.zeros((6, 2))
    dA, dB = grads(W, A, B, 4.0, 2, target)
    close(dA, np.zeros_like(A), rtol=0, atol=1e-12, label="dA with B = 0")
    if float(np.abs(dB).max()) < 1e-6:
        fail("dB should be nonzero when B = 0 and A is nonzero, otherwise nothing can learn")


@check("One small step moves W_eff by -eta (alpha/r)^2 (G A^T A + B B^T G)")
def test_step_direction(sub):
    grads = has(sub, "lora_gradients")
    merge = has(sub, "merge_lora")
    W, A, B, target = _problem(7)
    alpha, r = 4.0, 2
    s = alpha / r
    before = merge(W, A, B, alpha, r)
    G = before - target
    dA, dB = grads(W, A, B, alpha, r, target)
    eta = 1e-4
    after = merge(W, A - eta * dA, B - eta * dB, alpha, r)
    want = -eta * s**2 * (G @ A.T @ A + B @ B.T @ G)
    close(after - before, want, rtol=1e-2, atol=1e-9, label="change in W_eff after one step")
