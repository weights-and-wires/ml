"""The training recipe for a causal transformer: loss, schedule, clipping, and one AdamW step.

Implement the six functions below. `causal_lm_loss`, `perplexity`,
`warmup_cosine_lr`, `clip_global_norm`, and `training_flops` are small, pure
pieces of arithmetic. `train_step` wires them together with PyTorch's autograd
to run one step of training on any differentiable model.

The `__main__` block below trains a tiny character-level transformer, built
from plain torch operations, on a short repeating string. Everything in this
file above `train_step`'s call site is scaffolding: the model architecture,
its parameter initialisation, and a greedy sampler. None of it is part of
what you implement, and none of it needs to change.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# The six functions the checks call. Fill these in.
# ---------------------------------------------------------------------------


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy loss, in nats.

    Compute log-softmax from `logits` directly (subtract the log-sum-exp along
    the last axis) rather than calling a built-in cross-entropy function, then
    gather the log-probability of the true class at every position and return
    the negative mean.

    Args:
        logits: Unnormalised scores, shape (..., V).
        targets: Integer class indices, shape matching logits without the
            last axis, i.e. logits.shape[:-1].

    Returns:
        A scalar tensor: the mean negative log-probability of the targets,
        differentiable with respect to logits.
    """
    raise NotImplementedError("causal_lm_loss")


def perplexity(loss: float) -> float:
    """Return exp(loss), the exponentiated mean cross-entropy.

    Args:
        loss: The mean cross-entropy in nats, as a plain float.

    Returns:
        The perplexity, as a plain float.
    """
    raise NotImplementedError("perplexity")


def warmup_cosine_lr(step: float, warmup: float, total: float, peak: float, floor: float) -> float:
    """Linear warmup into cosine decay.

    Below `warmup`, the learning rate rises linearly from 0 to `peak`. From
    `warmup` to `total`, it decays along a cosine from `peak` down to `floor`,
    reaching `floor` exactly at step `total`. The two pieces agree exactly at
    step == warmup.

    Args:
        step: The current step, zero-indexed. May be a float.
        warmup: The number of warmup steps.
        total: The total number of steps in the schedule, total > warmup.
        peak: The learning rate at the end of warmup.
        floor: The learning rate at step == total.

    Returns:
        The learning rate at `step`, as a float.
    """
    raise NotImplementedError("warmup_cosine_lr")


def clip_global_norm(grads: list, max_norm: float) -> float:
    """Rescale a list of gradient arrays in place to a global norm cap.

    Compute the norm across every array at once, the square root of the sum
    of squared norms, exactly like `numpy.linalg.norm` applied to every array
    flattened and concatenated. When that total exceeds `max_norm`, multiply
    every array in the list, in place, by the same scalar `max_norm /
    total_norm`, so their new global norm is exactly `max_norm`. When the
    total does not exceed `max_norm`, leave every array unchanged.

    Works on a list of NumPy arrays or a list of PyTorch tensors.

    Args:
        grads: A list of arrays (or tensors) of arbitrary, possibly differing
            shapes.
        max_norm: The cap on the global norm.

    Returns:
        The global norm before any clipping, as a float.
    """
    raise NotImplementedError("clip_global_norm")


def training_flops(params: int, tokens: int) -> float:
    """Return the 6ND estimate of total training compute.

    Args:
        params: The model's non-embedding parameter count, N.
        tokens: The number of training tokens processed, D.

    Returns:
        Approximately 6 * params * tokens.
    """
    raise NotImplementedError("training_flops")


def train_step(model_fn, params: dict, batch: dict, opt_state: dict, config: dict):
    """Run one training step: forward, loss, backward, clip, AdamW update.

    Compute `logits = model_fn(params, batch["inputs"])`, the loss against
    `batch["targets"]` with `causal_lm_loss`, and call `.backward()` on it to
    populate each parameter's `.grad`. Clip the gradients as one flock with
    `clip_global_norm` and `config["max_grad_norm"]`. Advance the step count
    by one, look up the learning rate for the new step with
    `warmup_cosine_lr` and `config`'s schedule fields, and apply the
    bias-corrected AdamW update to every parameter:

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        theta = theta * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

    with `weight_decay` set to 0 for any parameter name in
    `config["no_decay"]`. Return brand-new leaf tensors (created with
    `.detach().requires_grad_()`) rather than mutating `params` in place, so
    the returned params carry no gradient history from this step.

    Args:
        model_fn: A callable `model_fn(params, input_ids) -> logits`, built
            from differentiable torch operations on the tensors in `params`.
        params: Dict of str to torch.Tensor, each a leaf tensor with
            `requires_grad=True`.
        batch: Dict with "inputs" and "targets", both integer tensors of
            shape (B, T), `targets` the sequence shifted by one position.
        opt_state: Dict with "step" (int), "m" and "v" (each a dict of str to
            torch.Tensor, matching `params`'s keys and shapes).
        config: Dict with "peak_lr", "floor_lr", "warmup_steps",
            "total_steps", "beta1", "beta2", "eps", "weight_decay",
            "max_grad_norm", and "no_decay" (a set or list of parameter
            names to exclude from weight decay).

    Returns:
        A tuple (new_params, new_opt_state, loss), where loss is a plain
        float.
    """
    raise NotImplementedError("train_step")


# ---------------------------------------------------------------------------
# Scaffolding for the __main__ demo: a tiny transformer built from plain
# torch operations. Nothing below this line is part of the exercise.
# ---------------------------------------------------------------------------


def _layer_norm(x, weight, bias, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def _causal_mask(T: int) -> torch.Tensor:
    mask = torch.zeros(T, T)
    mask.masked_fill_(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))
    return mask


def _attention(x, Wq, Wk, Wv, Wo, n_heads: int, mask):
    T, d_model = x.shape
    d_head = d_model // n_heads
    q = (x @ Wq).view(T, n_heads, d_head).transpose(0, 1)
    k = (x @ Wk).view(T, n_heads, d_head).transpose(0, 1)
    v = (x @ Wv).view(T, n_heads, d_head).transpose(0, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_head) + mask
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ v).transpose(0, 1).reshape(T, d_model)
    return out @ Wo


def _mlp(x, W1, b1, W2, b2):
    return torch.relu(x @ W1 + b1) @ W2 + b2


def _init_params(vocab_size, d_model, n_heads, n_layers, block_size, seed=0):
    g = torch.Generator().manual_seed(seed)

    def small(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).requires_grad_()

    params = {
        "embed": small(vocab_size, d_model),
        "pos_embed": small(block_size, d_model),
        "ln_f_w": torch.ones(d_model, requires_grad=True),
        "ln_f_b": torch.zeros(d_model, requires_grad=True),
    }
    for layer in range(n_layers):
        p = f"layer{layer}."
        params[p + "Wq"] = small(d_model, d_model)
        params[p + "Wk"] = small(d_model, d_model)
        params[p + "Wv"] = small(d_model, d_model)
        params[p + "Wo"] = small(d_model, d_model)
        params[p + "ln1_w"] = torch.ones(d_model, requires_grad=True)
        params[p + "ln1_b"] = torch.zeros(d_model, requires_grad=True)
        params[p + "ln2_w"] = torch.ones(d_model, requires_grad=True)
        params[p + "ln2_b"] = torch.zeros(d_model, requires_grad=True)
        params[p + "W1"] = small(d_model, 4 * d_model)
        params[p + "b1"] = torch.zeros(4 * d_model, requires_grad=True)
        params[p + "W2"] = small(4 * d_model, d_model)
        params[p + "b2"] = torch.zeros(d_model, requires_grad=True)
    return params


def _no_decay_names(params) -> set:
    return {name for name, value in params.items() if value.dim() < 2}


def _make_model_fn(n_layers: int, n_heads: int):
    def model_fn(params, input_ids):
        B, T = input_ids.shape
        mask = _causal_mask(T)
        outputs = []
        for b in range(B):
            x = params["embed"][input_ids[b]] + params["pos_embed"][:T]
            for layer in range(n_layers):
                p = f"layer{layer}."
                attn_in = _layer_norm(x, params[p + "ln1_w"], params[p + "ln1_b"])
                x = x + _attention(
                    attn_in,
                    params[p + "Wq"], params[p + "Wk"], params[p + "Wv"], params[p + "Wo"],
                    n_heads, mask,
                )
                mlp_in = _layer_norm(x, params[p + "ln2_w"], params[p + "ln2_b"])
                x = x + _mlp(mlp_in, params[p + "W1"], params[p + "b1"], params[p + "W2"], params[p + "b2"])
            x = _layer_norm(x, params["ln_f_w"], params["ln_f_b"])
            outputs.append(x @ params["embed"].T)
        return torch.stack(outputs, dim=0)

    return model_fn


def _generate(model_fn, params, stoi, itos, prompt: str, length: int, block_size: int) -> str:
    ids = [stoi[c] for c in prompt]
    for _ in range(length):
        context = ids[-block_size:]
        x = torch.tensor([context], dtype=torch.long)
        with torch.no_grad():
            logits = model_fn(params, x)
        next_id = int(torch.argmax(logits[0, -1]))
        ids.append(next_id)
    return "".join(itos[i] for i in ids)


if __name__ == "__main__":
    torch.manual_seed(0)
    text = "to be or not to be, that is the question. " * 30
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    vocab_size = len(chars)
    d_model, n_heads, n_layers, block_size = 32, 4, 2, 16
    batch_size, total_steps = 16, 300

    params = _init_params(vocab_size, d_model, n_heads, n_layers, block_size, seed=0)
    model_fn = _make_model_fn(n_layers, n_heads)
    no_decay = _no_decay_names(params)
    n_params = sum(v.numel() for v in params.values())

    opt_state = {
        "step": 0,
        "m": {k: torch.zeros_like(v) for k, v in params.items()},
        "v": {k: torch.zeros_like(v) for k, v in params.items()},
    }
    config = {
        "peak_lr": 3e-3,
        "floor_lr": 3e-4,
        "warmup_steps": 30,
        "total_steps": total_steps,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "no_decay": no_decay,
    }

    rng = np.random.default_rng(0)
    for step in range(total_steps):
        starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        inputs = torch.stack([data[s : s + block_size] for s in starts])
        targets = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
        batch = {"inputs": inputs, "targets": targets}

        params, opt_state, loss = train_step(model_fn, params, batch, opt_state, config)

        if step % 50 == 0 or step == total_steps - 1:
            print(f"step {step:4d}  loss {loss:.3f}  perplexity {perplexity(loss):.2f}")

    tokens_seen = total_steps * batch_size * block_size
    print(f"training flops (this run): {training_flops(n_params, tokens_seen):.3e}")
    sample = _generate(model_fn, params, stoi, itos, prompt="to be ", length=40, block_size=block_size)
    print(f"sample: {sample!r}")
