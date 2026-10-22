"""Checks for the merging exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has


@check("merge_lora gives [[3, 4], [0, 1]] on the chapter's 2 by 2 example")
def test_hand_example(sub):
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


@check("merged_matches_split is also tiny in 32-bit arithmetic")
def test_gap_float32(sub):
    gap_fn = has(sub, "merged_matches_split")
    rng = np.random.default_rng(4)
    d_in, d_out, r = 16, 10, 2
    x = rng.normal(size=(4, d_in)).astype(np.float32)
    W = (rng.normal(size=(d_out, d_in)) * 0.1).astype(np.float32)
    A = rng.normal(size=(r, d_in)).astype(np.float32)
    B = rng.normal(size=(d_out, r)).astype(np.float32)
    gap = float(gap_fn(x, W, A, B, 2.0, r))
    if not gap < 1e-4:
        fail(f"in float32 the gap should still be below 1e-4, got {gap:.3e}")
