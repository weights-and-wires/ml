"""Checks for the parameter-efficient fine-tuning exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has


@check("With B initialised to zero, lora_forward exactly reproduces the base layer")
def test_zero_init(sub):
    got = has(sub, "lora_forward")
    rng = np.random.default_rng(0)
    d_in, d_out, r, alpha = 10, 8, 3, 4.0
    x = rng.normal(size=(6, d_in))
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = np.zeros((d_out, r))
    close(got(x, W, A, B, alpha, r), x @ W.T, rtol=0, atol=0, label="lora_forward with B=0")


@check("lora_parameter_count matches the hand-computed r(d_in + d_out)")
def test_parameter_count(sub):
    got = has(sub, "lora_parameter_count")
    for d_in, d_out, r in [(4096, 4096, 8), (512, 2048, 16), (128, 128, 1)]:
        want = r * (d_in + d_out)
        value = got(d_in, d_out, r)
        if value != want:
            fail(
                f"lora_parameter_count({d_in}, {d_out}, {r}) is {value:,}, "
                f"expected {want:,}"
            )


@check("optimiser_memory_bytes charges 16 bytes per parameter, and the saving matches the ratio")
def test_optimiser_memory(sub):
    memory = has(sub, "optimiser_memory_bytes")
    count = has(sub, "lora_parameter_count")
    d_in, d_out, r = 4096, 4096, 8

    full_params = d_in * d_out
    lora_params = count(d_in, d_out, r)

    full_bytes = memory(full_params, "full")
    lora_bytes = memory(lora_params, "lora")
    if full_bytes != 16 * full_params:
        fail(f"optimiser_memory_bytes(full) is {full_bytes:,}, expected {16 * full_params:,}")
    if lora_bytes != 16 * lora_params:
        fail(f"optimiser_memory_bytes(lora) is {lora_bytes:,}, expected {16 * lora_params:,}")

    ratio = full_bytes / lora_bytes
    want_ratio = full_params / lora_params
    if abs(ratio - want_ratio) > 1e-6 * want_ratio:
        fail(
            f"the optimizer-memory ratio should equal the parameter-count ratio "
            f"({want_ratio:.3f}), since both modes use the same rate, but got {ratio:.3f}"
        )


@check("merge_lora gives outputs identical to lora_forward, to machine precision")
def test_merge_matches_forward(sub):
    forward = has(sub, "lora_forward")
    merge = has(sub, "merge_lora")
    rng = np.random.default_rng(1)
    d_in, d_out, r, alpha = 12, 9, 3, 6.0
    x = rng.normal(size=(7, d_in))
    W = rng.normal(size=(d_out, d_in)) * 0.3
    A = rng.normal(size=(r, d_in)) * 0.3
    B = rng.normal(size=(d_out, r)) * 0.3

    y_forward = forward(x, W, A, B, alpha, r)
    y_merged = x @ merge(W, A, B, alpha, r).T
    close(y_merged, y_forward, rtol=1e-10, atol=1e-12, label="merged weight vs. lora_forward")


@check("best_rank_approximation achieves the Eckart-Young error for the chosen rank")
def test_eckart_young(sub):
    got = has(sub, "best_rank_approximation")
    rng = np.random.default_rng(2)
    m, n, r = 20, 14, 4
    u, _ = np.linalg.qr(rng.normal(size=(m, m)))
    v, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.array([10.0, 7.0, 4.0, 2.5, 1.2, 0.8, 0.3][: min(m, n)])
    if s.size < min(m, n):
        s = np.pad(s, (0, min(m, n) - s.size))
    delta_w = (u[:, : s.size] * s) @ v[:, : s.size].T

    approx = got(delta_w, r)
    error = float(np.linalg.norm(delta_w - approx, ord="fro"))
    want = float(np.sqrt(np.sum(s[r:] ** 2)))
    close(error, want, rtol=1e-6, atol=1e-8, label="Frobenius error at rank r")


@check("The gradient of merge_lora's effective weight matches finite differences")
def test_gradient_matches_finite_difference(sub):
    merge = has(sub, "merge_lora")
    rng = np.random.default_rng(5)
    d_in, d_out, r, alpha = 6, 5, 2, 4.0
    W = rng.normal(size=(d_out, d_in)) * 0.1
    A = rng.normal(size=(r, d_in)) * 0.1
    B = rng.normal(size=(d_out, r)) * 0.1
    target = rng.normal(size=(d_out, d_in)) * 0.1

    def loss_of_a(a):
        w_eff = merge(W, a, B, alpha, r)
        return 0.5 * float(np.sum((w_eff - target) ** 2))

    def loss_of_b(b):
        w_eff = merge(W, A, b, alpha, r)
        return 0.5 * float(np.sum((w_eff - target) ** 2))

    g = merge(W, A, B, alpha, r) - target
    analytic_da = (alpha / r) * (B.T @ g)
    analytic_db = (alpha / r) * (g @ A.T)

    numeric_da = finite_difference(loss_of_a, A, eps=1e-5)
    numeric_db = finite_difference(loss_of_b, B, eps=1e-5)

    close(numeric_da, analytic_da, rtol=1e-3, atol=1e-5, label="dL/dA vs. finite differences")
    close(numeric_db, analytic_db, rtol=1e-3, atol=1e-5, label="dL/dB vs. finite differences")


@check("A gradient step on A and B moves W_eff along the projected full-fine-tuning direction")
def test_projected_gradient_direction(sub):
    merge = has(sub, "merge_lora")
    rng = np.random.default_rng(6)
    d_in, d_out, r, alpha = 7, 6, 2, 4.0
    W = rng.normal(size=(d_out, d_in)) * 0.1
    A = rng.normal(size=(r, d_in)) * 0.1
    B = rng.normal(size=(d_out, r)) * 0.1
    target = rng.normal(size=(d_out, d_in)) * 0.1

    w_before = merge(W, A, B, alpha, r)
    g = w_before - target
    da = (alpha / r) * (B.T @ g)
    db = (alpha / r) * (g @ A.T)

    eta = 1e-4
    w_after = merge(W, A - eta * da, B - eta * db, alpha, r)
    actual_delta = w_after - w_before

    predicted_delta = -eta * (alpha / r) ** 2 * (g @ A.T @ A + B @ B.T @ g)

    close(
        actual_delta,
        predicted_delta,
        rtol=1e-2,
        atol=1e-8,
        label="change in W_eff vs. the projected full-fine-tuning gradient",
    )
