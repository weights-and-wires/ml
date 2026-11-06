"""Checks for the LoRA exercise."""

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


@check("merge_lora gives [[3, 4], [0, 1]] on the chapter's 2 by 2 example")
def test_hand_example_merge(sub):
    merge = has(sub, "merge_lora")
    W = np.eye(2)
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0], [0.0]])
    close(merge(W, A, B, 2.0, 1), np.array([[3.0, 4.0], [0.0, 1.0]]), label="merged weight")


@check("merge_lora returns shape (d_out, d_in) on a non-square layer and equals W + (alpha / r) B A")
def test_shape(sub):
    merge = has(sub, "merge_lora")
    rng = np.random.default_rng(0)
    d_in, d_out, r, alpha = 8, 5, 3, 6.0
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = rng.normal(size=(d_out, r))
    got = merge(W, A, B, alpha, r)
    equal(np.shape(got), (d_out, d_in), label="shape of the merged weight")
    close(got, W + (alpha / r) * (B @ A), rtol=1e-12, atol=1e-12, label="merged weight")


@check("Merging changes W by a matrix of rank at most r")
def test_rank(sub):
    merge = has(sub, "merge_lora")
    rng = np.random.default_rng(1)
    d_in, d_out, r = 15, 12, 2
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = rng.normal(size=(d_out, r))
    rank = int(np.linalg.matrix_rank(merge(W, A, B, 4.0, r) - W))
    if rank != r:
        fail(f"merged minus W should have rank {r}, got rank {rank}")


@check("x @ merged.T matches the split path from lora_forward")
def test_outputs_match(sub):
    merge = has(sub, "merge_lora")
    forward = has(sub, "lora_forward")
    rng = np.random.default_rng(2)
    d_in, d_out, r, alpha = 12, 9, 3, 6.0
    x = rng.normal(size=(7, d_in))
    W = rng.normal(size=(d_out, d_in)) * 0.3
    A = rng.normal(size=(r, d_in)) * 0.3
    B = rng.normal(size=(d_out, r)) * 0.3
    close(x @ merge(W, A, B, alpha, r).T, forward(x, W, A, B, alpha, r),
          rtol=1e-10, atol=1e-12, label="merged vs split outputs")


@check("extra_multiplies returns r * (d_in + d_out)")
def test_extra_multiplies(sub):
    extra = has(sub, "extra_multiplies")
    for d_in, d_out, r, want in [(4096, 4096, 8, 65_536), (512, 2048, 16, 40_960), (2, 2, 1, 4)]:
        got = extra(d_in, d_out, r)
        if got != want:
            fail(f"extra_multiplies({d_in}, {d_out}, {r}) is {got:,}, expected {want:,}")
    share = extra(4096, 4096, 8) / (4096 * 4096)
    close(share, 1 / 256, rtol=1e-12, atol=0, label="share of the base layer's multiplies")


@check("merged_matches_split is tiny for correct code and returns a number")
def test_gap(sub):
    gap_fn = has(sub, "merged_matches_split")
    rng = np.random.default_rng(3)
    d_in, d_out, r, alpha = 20, 14, 4, 8.0
    x = rng.normal(size=(6, d_in))
    W = rng.normal(size=(d_out, d_in))
    A = rng.normal(size=(r, d_in))
    B = rng.normal(size=(d_out, r))
    gap = gap_fn(x, W, A, B, alpha, r)
    if np.ndim(gap) != 0:
        fail(f"merged_matches_split should return one number, got an array of shape {np.shape(gap)}")
    if not 0.0 <= float(gap) < 1e-10:
        fail(f"the merged and split outputs should agree to about 1e-12, but the gap is {float(gap):.3e}")
