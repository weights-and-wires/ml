"""Checks for the low-rank reach and training exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has


def _problem(seed, d_in=7, d_out=6, r=2):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d_out, d_in)) * 0.1
    A = rng.normal(size=(r, d_in)) * 0.3
    B = rng.normal(size=(d_out, r)) * 0.3
    target = rng.normal(size=(d_out, d_in)) * 0.3
    return W, A, B, target


@check("best_rank_approximation of [[2, 1], [1, 2]] at rank 1 is the all-1.5 matrix")
def test_hand_example(sub):
    approx = has(sub, "best_rank_approximation")
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    close(approx(M, 1), np.full((2, 2), 1.5), rtol=1e-9, atol=1e-9, label="rank-1 approximation")


@check("best_rank_approximation has rank r and the Eckart-Young error")
def test_eckart_young(sub):
    approx = has(sub, "best_rank_approximation")
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


@check("best_rank_approximation is closer than the smallest-values choice and than random rank-r matrices")
def test_beats_competitors(sub):
    approx = has(sub, "best_rank_approximation")
    rng = np.random.default_rng(3)
    delta_w = rng.normal(size=(12, 9))
    r = 3
    best = float(np.linalg.norm(delta_w - approx(delta_w, r)))
    U, s, Vt = np.linalg.svd(delta_w, full_matrices=False)
    worst = (U[:, -r:] * s[-r:]) @ Vt[-r:]
    if best >= float(np.linalg.norm(delta_w - worst)):
        fail("keeping the smallest singular values should be worse than your approximation")
    for _ in range(20):
        rand = rng.normal(size=(12, r)) @ rng.normal(size=(r, 9))
        if best > float(np.linalg.norm(delta_w - rand)):
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


@check("lora_gradients gives dA = [[4, 6]] and dB = [[16], [0]] on the chapter's example")
def test_hand_gradients(sub):
    grads = has(sub, "lora_gradients")
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    target = np.array([[1.0, 1.0], [2.0, 0.0]])
    dA, dB = grads(W, A, B, 2.0, 1, target)
    close(dA, np.array([[4.0, 6.0]]), label="dA")
    close(dB, np.array([[16.0], [0.0]]), label="dB")


@check("lora_gradients matches finite differences of 0.5 * ||W_eff - target||^2")
def test_finite_difference(sub):
    grads = has(sub, "lora_gradients")
    merge = has(sub, "merge_lora")
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


@check("With B = 0, the gradient for A is zero and the gradient for B is not")
def test_zero_b(sub):
    grads = has(sub, "lora_gradients")
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
