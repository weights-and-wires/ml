"""Checks for the LoRA layer exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has


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


@check("lora_init is repeatable for one seed and different for another")
def test_init_seed(sub):
    init = has(sub, "lora_init")
    A1, _ = init(10, 6, 3, 7)
    A2, _ = init(10, 6, 3, 7)
    A3, _ = init(10, 6, 3, 8)
    close(A1, A2, rtol=0, atol=0, label="same seed")
    if np.allclose(A1, A3):
        fail("different seeds gave the same A")


@check("With B at zero, lora_forward reproduces the frozen layer")
def test_zero_b(sub):
    forward = has(sub, "lora_forward")
    rng = np.random.default_rng(0)
    d_in, d_out, r = 10, 8, 3
    x = rng.normal(size=(6, d_in))
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = np.zeros((d_out, r))
    close(forward(x, W, A, B, 4.0, r), x @ W.T, rtol=1e-12, atol=1e-12, label="B = 0 output")


@check("lora_forward gives [7, 1] on the chapter's 2 by 2 example")
def test_hand_example(sub):
    forward = has(sub, "lora_forward")
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    x = np.array([[1.0, 1.0]])
    close(forward(x, W, A, B, 2.0, 1), np.array([[7.0, 1.0]]), label="hand example")


@check("lora_forward equals x @ (W + (alpha / r) B A).T on random non-square inputs")
def test_matches_effective_weight(sub):
    forward = has(sub, "lora_forward")
    rng = np.random.default_rng(1)
    d_in, d_out, r, alpha = 12, 9, 3, 6.0
    x = rng.normal(size=(7, d_in))
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = rng.normal(size=(d_out, r))
    want = x @ (W + (alpha / r) * (B @ A)).T
    close(forward(x, W, A, B, alpha, r), want, rtol=1e-10, atol=1e-10, label="effective weight")


@check("The correction lora_forward adds has rank r, never more")
def test_update_rank(sub):
    forward = has(sub, "lora_forward")
    rng = np.random.default_rng(2)
    d_in, d_out, r = 14, 11, 3
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = rng.normal(size=(d_out, r))
    eye = np.eye(d_in)
    update = forward(eye, W, A, B, 3.0, r) - eye @ W.T
    rank = int(np.linalg.matrix_rank(update))
    if rank != r:
        fail(f"the added correction should have rank {r}, got rank {rank}")
