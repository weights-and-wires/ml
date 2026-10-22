"""Checks for the safe steps exercise."""

import math

import numpy as np
import torch

from mlcheck import almost, check, close, fail, has


@check("warmup_cosine_lr hits the four worked-example points")
def test_schedule_points(sub):
    got = has(sub, "warmup_cosine_lr")
    args = (10, 50, 2.0, 0.1)
    almost(got(5, *args), 1.0, tol=1e-9, label="step 5, in the middle of warmup")
    almost(got(10, *args), 2.0, tol=1e-9, label="step 10, the peak")
    almost(got(30, *args), 1.05, tol=1e-9, label="step 30, halfway down the glide")
    almost(got(50, *args), 0.1, tol=1e-9, label="step 50, the floor")


@check("The schedule starts at zero, never exceeds the peak, and the two pieces meet without a jump")
def test_schedule_shape(sub):
    got = has(sub, "warmup_cosine_lr")
    args = (10, 50, 2.0, 0.1)
    almost(got(0, *args), 0.0, tol=1e-12, label="step 0")
    left = got(10 - 1e-7, *args)
    right = got(10 + 1e-7, *args)
    if abs(left - right) > 1e-5:
        fail(f"the schedule jumps at the end of warmup: {left:.6f} just before, {right:.6f} just after")
    values = [got(s, *args) for s in range(0, 51)]
    if max(values) > 2.0 + 1e-12:
        fail(f"the schedule rises above the peak: {max(values)}")
    up = values[:11]
    down = values[10:]
    if any(b < a - 1e-12 for a, b in zip(up, up[1:])):
        fail("the rate should only rise during warmup")
    if any(b > a + 1e-12 for a, b in zip(down, down[1:])):
        fail("the rate should only fall after warmup")
    if min(values[10:]) < 0.1 - 1e-12:
        fail("the rate falls below the floor")


@check("global_norm measures every entry of every array as one vector")
def test_global_norm(sub):
    got = has(sub, "global_norm")
    almost(got([np.array([3.0, 4.0]), np.array([0.0])]), 5.0, tol=1e-12, label="the worked example")
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(3, 4)), rng.normal(size=(5,)), rng.normal(size=(2, 2, 2))]
    flat = np.concatenate([a.ravel() for a in arrays])
    keep = [a.copy() for a in arrays]
    almost(got(arrays), float(np.linalg.norm(flat)), tol=1e-10, label="mixed shapes")
    for a, k in zip(arrays, keep):
        close(a, k, atol=0, label="global_norm must not change the arrays")
    tensors = [torch.tensor(a) for a in arrays]
    almost(got(tensors), float(np.linalg.norm(flat)), tol=1e-10, label="torch tensors")
    if not isinstance(got(arrays), float):
        fail(f"global_norm should return a plain float, got {type(got(arrays)).__name__}")


@check("clip_global_norm leaves a short gradient alone and returns its norm")
def test_clip_noop(sub):
    got = has(sub, "clip_global_norm")
    grads = [np.array([3.0, 4.0]), np.array([0.0])]
    reported = got(grads, max_norm=10.0)
    almost(reported, 5.0, tol=1e-12, label="reported norm")
    close(grads[0], np.array([3.0, 4.0]), atol=0, label="gradient under the limit")


@check("clip_global_norm shortens the gradient to the limit and keeps its direction")
def test_clip_scales(sub):
    got = has(sub, "clip_global_norm")
    grads = [np.array([3.0, 4.0]), np.array([0.0])]
    reported = got(grads, max_norm=2.5)
    almost(reported, 5.0, tol=1e-12, label="norm reported from before the clip")
    close(grads[0], np.array([1.5, 2.0]), atol=1e-12, label="the worked example, [3, 4] becomes [1.5, 2]")

    rng = np.random.default_rng(1)
    arrays = [rng.normal(size=(4, 3)) * 10, rng.normal(size=(6,)) * 10]
    before = [a.copy() for a in arrays]
    flat_before = np.concatenate([a.ravel() for a in before])
    got(arrays, max_norm=1.0)
    flat_after = np.concatenate([a.ravel() for a in arrays])
    almost(float(np.linalg.norm(flat_after)), 1.0, tol=1e-9, label="norm after the clip")
    close(
        flat_after / np.linalg.norm(flat_after),
        flat_before / np.linalg.norm(flat_before),
        atol=1e-12,
        label="direction after the clip",
    )


@check("clip_global_norm changes torch gradients in place")
def test_clip_in_place(sub):
    got = has(sub, "clip_global_norm")
    a = torch.tensor([3.0, 4.0], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    original_a = a
    reported = got([a, b], max_norm=2.5)
    almost(reported, 5.0, tol=1e-12, label="norm reported from before the clip")
    if a is not original_a or not torch.allclose(a, torch.tensor([1.5, 2.0], dtype=torch.float64)):
        fail(f"the caller's tensor should now hold [1.5, 2.0], but holds {a.tolist()}; scale with g *= scale")


@check("A spike is cut down to the limit while ordinary gradients pass untouched")
def test_spike(sub):
    got = has(sub, "clip_global_norm")
    rng = np.random.default_rng(2)
    base = [rng.normal(size=(5, 5)) * 0.1, rng.normal(size=(5,)) * 0.1]
    ordinary = [a.copy() for a in base]
    n = got(ordinary, max_norm=1.0)
    if n > 1.0:
        fail(f"the test gradient should have norm under 1, but has {n}")
    for a, b in zip(ordinary, base):
        close(a, b, atol=0, label="an ordinary gradient must not change")
    spike = [a * 100 for a in base]
    n_spike = got(spike, max_norm=1.0)
    almost(n_spike, 100 * n, tol=1e-8, label="the spike's reported norm")
    almost(math.sqrt(sum(float((a**2).sum()) for a in spike)), 1.0, tol=1e-9, label="the spike after the clip")
