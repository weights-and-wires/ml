"""Checks for the training-a-transformer exercise."""

import math

import numpy as np
import torch

from mlcheck import almost, check, close, equal, fail, has


@check("causal_lm_loss on uniform logits equals log(V)")
def test_uniform_loss(sub):
    got = has(sub, "causal_lm_loss")
    V, T = 11, 6
    logits = torch.zeros(T, V, dtype=torch.float64)
    targets = torch.randint(0, V, (T,))
    loss = float(got(logits, targets))
    almost(loss, math.log(V), tol=1e-6, label="loss on uniform logits")

    # And the same, batched.
    logits_b = torch.zeros(3, T, V, dtype=torch.float64)
    targets_b = torch.randint(0, V, (3, T))
    loss_b = float(got(logits_b, targets_b))
    almost(loss_b, math.log(V), tol=1e-6, label="batched loss on uniform logits")


@check("causal_lm_loss matches a hand-computed cross-entropy on concentrated logits")
def test_loss_value(sub):
    got = has(sub, "causal_lm_loss")
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 1])

    def hand_cross_entropy(row, target):
        m = row.max()
        logsumexp = m + math.log(sum(math.exp(x - m) for x in row.tolist()))
        return logsumexp - row[target].item()

    want = (hand_cross_entropy(logits[0], 0) + hand_cross_entropy(logits[1], 1)) / 2
    almost(float(got(logits, targets)), want, tol=1e-8, label="mean cross-entropy")


@check("perplexity is the exponential of the loss")
def test_perplexity(sub):
    got = has(sub, "perplexity")
    almost(got(0.0), 1.0, tol=1e-9, label="perplexity of zero loss")
    almost(got(math.log(7.0)), 7.0, tol=1e-6, label="perplexity of log(7)")


@check("warmup_cosine_lr agrees at the warmup boundary and reaches its floor at the end")
def test_schedule(sub):
    got = has(sub, "warmup_cosine_lr")
    warmup, total, peak, floor = 10.0, 50.0, 2.0, 0.1

    almost(got(0.0, warmup, total, peak, floor), 0.0, tol=1e-9, label="start of warmup")
    almost(got(warmup, warmup, total, peak, floor), peak, tol=1e-9, label="warmup boundary")
    almost(got(total, warmup, total, peak, floor), floor, tol=1e-9, label="final step")

    left = got(warmup - 1e-6, warmup, total, peak, floor)
    right = got(warmup + 1e-6, warmup, total, peak, floor)
    if abs(left - right) > 1e-4:
        fail(
            "the schedule should be continuous across the warmup boundary, "
            f"but it jumps from {left:.6f} to {right:.6f}"
        )


@check("clip_global_norm rescales to exactly the threshold, and is a no-op below it")
def test_clip_global_norm(sub):
    got = has(sub, "clip_global_norm")

    grads = [np.array([3.0, 4.0]), np.array([0.0])]
    reported = got(grads, max_norm=10.0)
    almost(reported, 5.0, tol=1e-8, label="reported global norm")
    close(grads[0], np.array([3.0, 4.0]), atol=1e-12, label="unclipped gradient, below threshold")

    grads2 = [np.array([3.0, 4.0])]
    reported2 = got(grads2, max_norm=2.5)
    almost(reported2, 5.0, tol=1e-8, label="reported pre-clip norm")
    new_norm = float(np.sqrt(sum(float((g**2).sum()) for g in grads2)))
    almost(new_norm, 2.5, tol=1e-6, label="post-clip global norm")


@check("training_flops matches the 6ND hand computation")
def test_training_flops(sub):
    got = has(sub, "training_flops")
    equal(int(got(1_000_000, 10_000)), 6 * 1_000_000 * 10_000, label="training_flops")
    equal(int(got(357, 42)), 6 * 357 * 42, label="training_flops, small numbers")


@check("train_step applies one bias-corrected, decoupled-decay AdamW update")
def test_train_step_math(sub):
    train_step = has(sub, "train_step")

    embed0 = torch.eye(3, dtype=torch.float64, requires_grad=True)
    bias0 = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    params = {"embed": embed0, "bias": bias0}

    def model_fn(p, input_ids):
        return p["embed"][input_ids] + p["bias"]

    inputs = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 0]])
    batch = {"inputs": inputs, "targets": targets}

    opt_state = {
        "step": 0,
        "m": {k: torch.zeros_like(v) for k, v in params.items()},
        "v": {k: torch.zeros_like(v) for k, v in params.items()},
    }
    config = {
        "peak_lr": 0.1,
        "floor_lr": 0.01,
        "warmup_steps": 5,
        "total_steps": 20,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.1,
        "max_grad_norm": 1e6,  # large enough that clipping never triggers here
        "no_decay": {"bias"},
    }

    new_params, new_opt_state, loss = train_step(model_fn, params, batch, opt_state, config)
    equal(new_opt_state["step"], 1, label="opt_state step after one call")

    # Recompute the loss and the gradient independently.
    logits = model_fn(params, inputs)
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    almost(loss, float(nll), tol=1e-8, label="loss returned by train_step")
    nll.backward()
    grads = {"embed": params["embed"].grad.clone(), "bias": params["bias"].grad.clone()}

    lr = has(sub, "warmup_cosine_lr")(1, 5, 20, 0.1, 0.01)
    for name in ("embed", "bias"):
        g = grads[name]
        m = 0.1 * g
        v = 0.05 * g**2
        m_hat = m / (1 - 0.9**1)
        v_hat = v / (1 - 0.95**1)
        decay = 0.0 if name in config["no_decay"] else config["weight_decay"]
        want = params[name].detach() * (1 - lr * decay) - lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
        close(
            new_params[name].detach().numpy(),
            want.numpy(),
            atol=1e-8,
            label=f"updated {name!r}",
        )


@check("Twenty steps of train_step on a memorisable batch drive the loss well below log(V)")
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

    opt_state = {
        "step": 0,
        "m": {k: torch.zeros_like(v) for k, v in params.items()},
        "v": {k: torch.zeros_like(v) for k, v in params.items()},
    }
    config = {
        "peak_lr": 0.2,
        "floor_lr": 0.02,
        "warmup_steps": 5,
        "total_steps": 20,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "no_decay": set(),
    }

    loss = None
    for _ in range(20):
        params, opt_state, loss = train_step(model_fn, params, batch, opt_state, config)

    anchor = math.log(V)
    if loss > anchor * 0.5:
        fail(
            f"after 20 steps on a memorisable, repeating batch the loss is "
            f"{loss:.3f}, not clearly below the untrained anchor of "
            f"log(V) = {anchor:.3f}"
        )
