"""Checks for the next-token loss and safe steps exercise."""

import math

import numpy as np
import torch

from mlcheck import almost, check, close, equal, fail, has


@check("per_token_loss matches minus the log probability on the worked example, for any leading shape")
def test_worked_example(sub):
    got = has(sub, "per_token_loss")
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    out = got(logits, targets)
    equal(tuple(out.shape), (2,), label="shape of the per-token costs")
    close(out.detach().numpy(), np.array([0.2395, 0.0949]), atol=5e-5, label="per-token costs")

    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(3, 5, 7)))
    targets = torch.tensor(rng.integers(0, 7, size=(3, 5)))
    out = got(logits, targets)
    equal(tuple(out.shape), (3, 5), label="shape for (B, T, V) logits")
    z = logits.numpy()
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    want = -np.log(np.take_along_axis(p, targets.numpy()[..., None], -1)[..., 0])
    close(out.detach().numpy(), want, atol=1e-10, label="costs for (B, T, V) logits")


@check("per_token_loss stays finite and exact on logits of 1000")
def test_stability(sub):
    got = has(sub, "per_token_loss")
    logits = torch.tensor([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [1000.0, 999.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 1])
    out = got(logits, targets).detach().numpy()
    if not np.all(np.isfinite(out)):
        fail(f"per_token_loss returned non-finite values {out}; subtract the largest logit before exponentiating")
    want = np.array([0.0, 1000.0, 1.0 + math.log(1 + math.exp(-1))])
    close(out, want, atol=1e-8, label="costs at huge logits")


@check("The gradient of the mean loss is (softmax - one-hot) / N, confirmed by finite differences")
def test_gradient(sub):
    got = has(sub, "causal_lm_loss")
    rng = np.random.default_rng(1)
    z = rng.normal(size=(4, 5))
    t = torch.tensor(rng.integers(0, 5, size=4))

    def loss_at(arr):
        return float(got(torch.tensor(arr), t))

    numeric = np.zeros_like(z)
    h = 1e-6
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            up, down = z.copy(), z.copy()
            up[i, j] += h
            down[i, j] -= h
            numeric[i, j] = (loss_at(up) - loss_at(down)) / (2 * h)

    leaf = torch.tensor(z, requires_grad=True)
    got(leaf, t).backward()
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    onehot = np.eye(5)[t.numpy()]
    close(leaf.grad.numpy(), numeric, atol=1e-6, label="autograd against finite differences")
    close(leaf.grad.numpy(), (p - onehot) / 4, atol=1e-8, label="autograd against (softmax - one-hot) / N")


@check("perplexity reads the loss as a count of choices, and uniform logits score log V")
def test_perplexity(sub):
    per = has(sub, "per_token_loss")
    ppl = has(sub, "perplexity")
    almost(ppl(0.0), 1.0, tol=1e-12, label="a loss of 0 is one choice")
    almost(ppl(math.log(4.0)), 4.0, tol=1e-9, label="a loss of ln 4 is four choices")
    almost(ppl(torch.tensor(0.16723386)), 1.18203, tol=1e-4, label="the worked loss 0.1672")
    if not isinstance(ppl(0.5), float):
        fail(f"perplexity should return a plain float, got {type(ppl(0.5)).__name__}")
    for V in (11, 50_000):
        logits = torch.zeros(4, V, dtype=torch.float64)
        targets = torch.tensor([0, 1, 2, 3])
        loss = float(per(logits, targets).mean())
        almost(loss, math.log(V), tol=1e-9, label=f"loss on uniform logits, V = {V}")
        almost(ppl(loss), V, tol=1e-6 * V, label=f"perplexity, V = {V}")


@check("make_lm_batch shifts the targets one place left and refuses a window that runs past the end")
def test_batch(sub):
    got = has(sub, "make_lm_batch")
    tokens = torch.arange(20) * 3
    inputs, targets = got(tokens, 5, 3)
    close(np.asarray(inputs), np.array([9, 12, 15, 18, 21]), label="inputs")
    close(np.asarray(targets), np.array([12, 15, 18, 21, 24]), label="targets")
    inputs, targets = got(tokens, 19, 0)
    equal(len(inputs), 19, label="the longest window")
    close(np.asarray(targets)[-1], 57, label="the last target is the last token")
    try:
        got(torch.arange(10), 5, 5)
    except ValueError:
        return
    except Exception as err:  # noqa: BLE001
        fail(f"expected ValueError, got {type(err).__name__}: {err}")
    fail("start 5 with block_size 5 needs 11 tokens but there are 10; expected a ValueError")


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


@check("clip_global_norm leaves a short gradient alone, and cuts a spike to the limit")
def test_clip_noop_and_spike(sub):
    got = has(sub, "clip_global_norm")
    grads = [np.array([3.0, 4.0]), np.array([0.0])]
    reported = got(grads, max_norm=10.0)
    almost(reported, 5.0, tol=1e-12, label="reported norm")
    close(grads[0], np.array([3.0, 4.0]), atol=0, label="gradient under the limit")

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


@check("clip_global_norm shortens the gradient to the limit, keeps its direction, and works in place on torch tensors")
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

    a = torch.tensor([3.0, 4.0], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    original_a = a
    reported = got([a, b], max_norm=2.5)
    almost(reported, 5.0, tol=1e-12, label="norm reported from before the clip")
    if a is not original_a or not torch.allclose(a, torch.tensor([1.5, 2.0], dtype=torch.float64)):
        fail(f"the caller's tensor should now hold [1.5, 2.0], but holds {a.tolist()}; scale with g *= scale")
