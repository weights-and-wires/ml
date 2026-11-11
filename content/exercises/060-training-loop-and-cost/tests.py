"""Checks for the training loop and cost exercise."""

import math

import numpy as np
import torch

from mlcheck import almost, check, close, equal, fail, has

B1, B2, EPS = 0.9, 0.95, 1e-8


def _fresh_state(params):
    return {
        "step": 0,
        "m": {k: torch.zeros_like(v) for k, v in params.items()},
        "v": {k: torch.zeros_like(v) for k, v in params.items()},
    }


def test_second_step(sub):
    update = has(sub, "adamw_update")
    theta, m, v = update(
        np.array([0.89]), np.array([0.5]), np.array([0.05]), np.array([0.0125]), 2, 0.1, B1, B2, EPS, 0.1
    )
    close(m, np.array([0.095]), atol=1e-12, label="m after two steps")
    close(v, np.array([0.024375]), atol=1e-12, label="v after two steps")
    close(theta, np.array([0.7811]), atol=1e-6, label="weight after two steps")


def test_step_zero(sub):
    update = has(sub, "adamw_update")
    zero = np.zeros(1)
    for bad in (0, -1):
        try:
            update(np.ones(1), np.ones(1), zero, zero, bad, 0.1, B1, B2, EPS, 0.0)
        except ValueError:
            continue
        except Exception as err:  # noqa: BLE001
            fail(f"step {bad} should raise ValueError, got {type(err).__name__}: {err}")
        fail(f"step {bad} should raise ValueError, because 1 - beta ** {bad} is not a valid correction")


def _toy_problem():
    embed0 = torch.eye(3, dtype=torch.float64, requires_grad=True)
    bias0 = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    params = {"embed": embed0, "bias": bias0}

    def model_fn(p, input_ids):
        return p["embed"][input_ids] + p["bias"]

    batch = {"inputs": torch.tensor([[0, 1, 2, 0]]), "targets": torch.tensor([[1, 2, 0, 0]])}
    return params, model_fn, batch


def _config(**extra):
    config = {
        "peak_lr": 0.1,
        "floor_lr": 0.01,
        "warmup_steps": 5,
        "total_steps": 20,
        "beta1": B1,
        "beta2": B2,
        "eps": EPS,
        "weight_decay": 0.1,
        "max_grad_norm": 1e6,
        "no_decay": {"bias"},
    }
    config.update(extra)
    return config


def _raw_gradients(model_fn, params, batch):
    """The gradient of the mean loss, computed without touching `params`."""
    leaves = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
    logits = model_fn(leaves, batch["inputs"])
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    nll = -logp.gather(-1, batch["targets"].unsqueeze(-1)).squeeze(-1).mean()
    grads = torch.autograd.grad(nll, list(leaves.values()))
    return float(nll.detach()), dict(zip(leaves.keys(), grads))


def test_scale_invariance(sub):
    update = has(sub, "adamw_update")
    grad = np.array([0.3, -2.0, 15.0, -400.0, 0.05, 9000.0])
    zero = np.zeros(6)
    theta0 = np.ones(6)
    theta, _, _ = update(theta0, grad, zero, zero, 1, 0.01, B1, B2, EPS, 0.0)
    close(theta0 - theta, 0.01 * np.sign(grad), atol=1e-7, label="step lengths and signs")
    theta2, _, _ = update(theta0, grad * 1000, zero, zero, 1, 0.01, B1, B2, EPS, 0.0)
    close(theta2, theta, atol=1e-7, label="scaling every gradient by 1000 changes nothing")


@check("adamw_update reproduces the worked example, corrects the bias at step 2, and refuses step 0")
def test_worked_example(sub):
    update = has(sub, "adamw_update")
    zero = np.zeros(1)
    theta, m, v = update(np.array([1.0]), np.array([0.5]), zero, zero, 1, 0.1, B1, B2, EPS, 0.1)
    close(theta, np.array([0.89]), atol=1e-6, label="updated weight")
    close(m, np.array([0.05]), atol=1e-12, label="gradient average m, raw")
    close(v, np.array([0.0125]), atol=1e-12, label="squared-gradient average v, raw")
    test_second_step(sub)
    test_step_zero(sub)


@check("Weight decay shrinks a weight by lr * decay on its own, and every weight moves one learning rate, whatever its gradient size")
def test_decoupled_decay(sub):
    update = has(sub, "adamw_update")
    theta0 = np.array([2.0, -3.0, 0.5])
    zero = np.zeros(3)
    theta, m, v = update(theta0, zero, zero, zero, 1, 0.1, B1, B2, EPS, 0.5)
    close(theta, theta0 * (1 - 0.1 * 0.5), atol=1e-12, label="shrink alone, gradient zero")
    close(m, zero, atol=0, label="m stays zero")
    close(v, zero, atol=0, label="v stays zero")
    test_scale_invariance(sub)


@check("train_step matches an independent AdamW computation, with no decay on the bias")
def test_train_step_math(sub):
    train_step = has(sub, "train_step")
    params, model_fn, batch = _toy_problem()
    want_loss, raw = _raw_gradients(model_fn, params, batch)
    start = {k: v.detach().clone() for k, v in params.items()}
    new_params, new_state, loss = train_step(model_fn, params, batch, _fresh_state(params), _config())
    equal(new_state["step"], 1, label="step after one call")
    almost(loss, want_loss, tol=1e-9, label="loss returned by train_step")

    lr = 0.1 * 1 / 5  # step 1 of a 5-step warmup to a peak of 0.1
    for name in ("embed", "bias"):
        g = raw[name]
        m_hat = (0.1 * g) / (1 - B1)
        v_hat = (0.05 * g**2) / (1 - B2)
        decay = 0.0 if name == "bias" else 0.1
        want = start[name] * (1 - lr * decay) - lr * m_hat / (torch.sqrt(v_hat) + EPS)
        close(new_params[name].detach().numpy(), want.numpy(), atol=1e-8, label=f"updated {name!r}")
        close(new_state["m"][name].detach().numpy(), (0.1 * g).numpy(), atol=1e-10, label=f"stored m for {name!r}")
        close(new_state["v"][name].detach().numpy(), (0.05 * g**2).numpy(), atol=1e-10, label=f"stored v for {name!r}")


@check("train_step clips the gradients as one list before they reach the averages, and returns fresh leaves")
def test_clip_and_leaves(sub):
    train_step = has(sub, "train_step")
    params, model_fn, batch = _toy_problem()
    _, raw = _raw_gradients(model_fn, params, batch)
    raw_norm = math.sqrt(sum(float((g**2).sum()) for g in raw.values()))
    limit = 0.01
    if raw_norm < 10 * limit:
        fail("the toy gradient is unexpectedly small")

    start = {k: v.detach().clone() for k, v in params.items()}
    new_params, state, _ = train_step(model_fn, params, batch, _fresh_state(params), _config(max_grad_norm=limit))
    clipped = {k: state["m"][k] / 0.1 for k in start}
    clipped_norm = math.sqrt(sum(float((g**2).sum()) for g in clipped.values()))
    almost(clipped_norm, limit, tol=1e-9, label="global norm of the gradient the averages were built from")
    for k in start:
        close(clipped[k].detach().numpy(), (raw[k] * (limit / raw_norm)).numpy(), atol=1e-10,
              label=f"clipped gradient for {k!r}, scaled by the same factor as the rest")

    for k, p in new_params.items():
        if not (p.requires_grad and p.is_leaf):
            fail(f"returned {k!r} must be a leaf tensor with requires_grad=True")
        if p.grad is not None:
            fail(f"returned {k!r} should start with no gradient, so the next step does not add to this one")
        if p is params[k]:
            fail(f"returned {k!r} is the same object as the input; return a fresh tensor")
        close(params[k].detach().numpy(), start[k].numpy(), atol=0, label=f"input {k!r} must not change in place")


@check("Twenty steps of train_step on a memorisable batch drive the loss well below log V")
def test_convergence(sub):
    train_step = has(sub, "train_step")
    torch.manual_seed(0)
    V, d = 4, 8
    pattern = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
    batch = {"inputs": pattern[:-1].unsqueeze(0), "targets": pattern[1:].unsqueeze(0)}

    g = torch.Generator().manual_seed(0)
    params = {
        "embed": (torch.randn(V, d, generator=g) * 0.1).requires_grad_(),
        "head": (torch.randn(d, V, generator=g) * 0.1).requires_grad_(),
    }

    def model_fn(p, input_ids):
        return p["embed"][input_ids] @ p["head"]

    state = _fresh_state(params)
    config = _config(peak_lr=0.2, floor_lr=0.02, weight_decay=0.0, max_grad_norm=1.0, no_decay=set())
    loss = None
    for _ in range(20):
        params, state, loss = train_step(model_fn, params, batch, state, config)
    anchor = math.log(V)
    if loss > anchor * 0.5:
        fail(f"after 20 steps on a repeating batch the loss is {loss:.3f}, not clearly below log(V) = {anchor:.3f}")
def test_flops_scale(sub):
    got = has(sub, "training_flops")
    almost(float(got(10**9, 2 * 10**10)), 1.2e20, tol=1e8, label="1e9 weights on 2e10 tokens")
    almost(float(got(10**8, 10**10)), 6e18, tol=1e6, label="1e8 weights on 1e10 tokens")
    almost(float(got(2000, 300)) / float(got(1000, 300)), 2.0, tol=1e-12, label="doubling the weights doubles the cost")
    almost(float(got(1000, 600)) / float(got(1000, 300)), 2.0, tol=1e-12, label="doubling the tokens doubles the cost")


def test_memory_variants(sub):
    got = has(sub, "training_memory_bytes")
    equal(int(got(1_000_000, bytes_per_number=2)), 8_000_000, label="2-byte numbers")
    equal(int(got(1_000_000, copies=2)), 8_000_000, label="plain descent, two copies")
    equal(int(got(1_000_000, 2, 3)), 6_000_000, label="positional arguments")
    equal(int(got(3_000)), 3 * int(got(1_000)), label="linear in the weight count")


def test_scaled(sub):
    got = has(sub, "scaled_grad_survives")
    out = float(got(1e-8, 1024.0))
    if abs(out - 1e-8) > 1e-10:
        fail(f"1e-8 at scale 1024 came back as {out:.4e}, expected within 1 percent of 1e-8")
    arr = got(np.array([1e-8, 3e-8, 2.0]), 1024.0)
    equal(tuple(np.shape(arr)), (3,), label="the result keeps the shape of an array input")
    close(arr, np.array([1e-8, 3e-8, 2.0]), rtol=0.01, atol=0, label="an array of gradients")


def test_ordinary(sub):
    got = has(sub, "scaled_grad_survives")
    close(float(got(3e-4, 1024.0)), 3e-4, rtol=0.005, atol=0, label="3e-4 at scale 1024")
    close(float(got(0.37, 8.0)), 0.37, rtol=0.001, atol=0, label="0.37 at scale 8")


@check("training_flops is 6ND on the worked example and at the scale of a billion weights, and linear in both arguments")
def test_flops_worked(sub):
    got = has(sub, "training_flops")
    equal(int(got(1_000_000, 10_000)), 6 * 10**10, label="a million weights on ten thousand tokens")
    equal(int(got(357, 42)), 6 * 357 * 42, label="small numbers")
    test_flops_scale(sub)


@check("training_memory_bytes defaults to four fp32 copies, 16 bytes per weight, and responds to number size, copies, and weight count")
def test_memory_defaults(sub):
    got = has(sub, "training_memory_bytes")
    equal(int(got(1_000_000)), 16_000_000, label="a million weights")
    equal(int(got(10**9)), 16 * 10**9, label="a billion weights")
    test_memory_variants(sub)


@check("Loss scaling: 1e-8 vanishes in fp16 at scale 1, returns within 1 percent at 1024, and ordinary gradients are unharmed")
def test_vanishes(sub):
    got = has(sub, "scaled_grad_survives")
    almost(float(got(1e-8, 1.0)), 0.0, tol=0.0, label="1e-8 at scale 1")
    test_scaled(sub)
    test_ordinary(sub)


@check("A scale that is too large overflows to infinity")
def test_overflow(sub):
    got = has(sub, "scaled_grad_survives")
    out = float(got(1.0, 1e5))
    if not np.isinf(out):
        fail(f"1.0 at scale 1e5 is 1e5 in fp16, above the largest number 65504, so it should be infinity; got {out}")
    out2 = float(got(1e-3, 65536.0))
    if np.isinf(out2) or abs(out2 - 1e-3) > 1e-5:
        fail(f"1e-3 at scale 65536 is about 65.5, which fits in fp16, but came back as {out2}")
