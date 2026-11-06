"""Reference solution for the training loop exercise.

The loss, the schedule, and the clip from the earlier chapters are already in this file.

The `__main__` block trains a tiny character-level transformer, built from
plain torch operations, on a short repeating string. Everything below the
two functions you write is scaffolding: the model, its initialisation, and a
greedy sampler. None of it needs to change.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Given: functions you wrote in earlier chapters, already implemented.
# ---------------------------------------------------------------------------


# Written in "How a language model is scored: next-token loss and perplexity".
def per_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


# Written in "How a language model is scored: next-token loss and perplexity".
def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return per_token_loss(logits, targets).mean()


# Written in "How a language model is scored: next-token loss and perplexity".
def perplexity(loss) -> float:
    return math.exp(float(loss))


# Written in "How a language model is scored: next-token loss and perplexity".
def make_lm_batch(tokens, block_size: int, start: int):
    if start < 0 or start + block_size + 1 > len(tokens):
        raise ValueError("the window runs past the end of the tokens")
    return tokens[start : start + block_size], tokens[start + 1 : start + block_size + 1]


# Written in "Taking safe steps: warmup, cosine decay, and gradient clipping".
def warmup_cosine_lr(step: float, warmup: float, total: float, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


# Written in "Taking safe steps: warmup, cosine decay, and gradient clipping".
def global_norm(grads: list) -> float:
    return math.sqrt(sum(float((g**2).sum()) for g in grads))


# Written in "Taking safe steps: warmup, cosine decay, and gradient clipping".
def clip_global_norm(grads: list, max_norm: float) -> float:
    total = global_norm(grads)
    if total > max_norm:
        scale = max_norm / total
        for g in grads:
            g *= scale
    return total


# ---------------------------------------------------------------------------
# The two functions the checks call.
# ---------------------------------------------------------------------------


def adamw_update(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay):
    if step < 1:
        raise ValueError(f"step counts updates from 1, got {step}")
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    theta = theta * (1 - lr * weight_decay) - lr * m_hat / (v_hat**0.5 + eps)
    return theta, m, v


def train_step(model_fn, params: dict, batch: dict, opt_state: dict, config: dict):
    logits = model_fn(params, batch["inputs"])
    loss = causal_lm_loss(logits, batch["targets"])
    loss.backward()

    names = list(params.keys())
    clip_global_norm([params[name].grad for name in names], config["max_grad_norm"])

    step = opt_state["step"] + 1
    lr = warmup_cosine_lr(
        step,
        config["warmup_steps"],
        config["total_steps"],
        config["peak_lr"],
        config["floor_lr"],
    )
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.95)
    eps = config.get("eps", 1e-8)
    weight_decay = config.get("weight_decay", 0.0)
    no_decay = config.get("no_decay", set())

    new_params, new_m, new_v = {}, {}, {}
    with torch.no_grad():
        for name in names:
            decay = 0.0 if name in no_decay else weight_decay
            theta, m, v = adamw_update(
                params[name], params[name].grad, opt_state["m"][name], opt_state["v"][name],
                step, lr, beta1, beta2, eps, decay,
            )
            new_params[name] = theta.detach().requires_grad_()
            new_m[name] = m
            new_v[name] = v

    return new_params, {"step": step, "m": new_m, "v": new_v}, float(loss.item())


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
        pairs = [make_lm_batch(data, block_size, int(s)) for s in starts]
        batch = {
            "inputs": torch.stack([p[0] for p in pairs]),
            "targets": torch.stack([p[1] for p in pairs]),
        }

        params, opt_state, loss = train_step(model_fn, params, batch, opt_state, config)

        if step % 50 == 0 or step == total_steps - 1:
            print(f"step {step:4d}  loss {loss:.3f}  perplexity {perplexity(loss):.2f}")

    print(f"weights in the model: {n_params}")
    sample = _generate(model_fn, params, stoi, itos, prompt="to be ", length=40, block_size=block_size)
    print(f"sample: {sample!r}")
