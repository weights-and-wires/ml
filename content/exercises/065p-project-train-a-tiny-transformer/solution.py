"""Reference solution: train a tiny decoder-only transformer to reverse digit strings.

Sequences look like "483920>029384": six digits, a separator, then the same
six digits backwards. The helpers written in earlier chapters are given in full.
"""

import math

import numpy as np
import torch

torch.set_num_threads(1)

N_DIGITS = 6
SEP = 10  # token id of ">"; the digits 0-9 are their own ids
VOCAB = 11
SEQ_LEN = 2 * N_DIGITS + 1  # 13 tokens per example


# ---------------------------------------------------------------------------
# Given: the data, and helpers from earlier chapters.
# ---------------------------------------------------------------------------


def make_data(seed: int = 0, n_train: int = 3000, n_test: int = 300) -> dict:
    """Build train and held-out sequences. Every test string is absent from train."""
    rng = np.random.default_rng(seed)
    digits = rng.integers(0, 10, size=(n_train + 2 * n_test, N_DIGITS))
    keys = [tuple(r) for r in digits]
    seen, rows = set(), []
    for key, row in zip(keys, digits):
        if key not in seen:
            seen.add(key)
            rows.append(row)
    digits = np.array(rows)
    train_d, test_d = digits[:n_train], digits[n_train : n_train + n_test]

    def build(d):
        sep = np.full((len(d), 1), SEP)
        return torch.tensor(np.concatenate([d, sep, d[:, ::-1]], axis=1), dtype=torch.long)

    return {"train": build(train_d), "test": build(test_d)}


def per_token_loss(logits, targets):
    # Written in "How a language model is scored: next-token loss and perplexity".
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def layer_norm(x, gain, bias, eps: float = 1e-5):
    # Written in "Stacking it up: residuals, layer norm, and the pre-norm block".
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * gain + bias


def warmup_cosine_lr(step, warmup, total, peak, floor):
    # Written in "Taking safe steps: warmup, cosine decay, and gradient clipping".
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def clip_global_norm(grads, max_norm):
    # Written in "Taking safe steps: warmup, cosine decay, and gradient clipping".
    total = math.sqrt(sum(float((g**2).sum()) for g in grads))
    if total > max_norm:
        for g in grads:
            g *= max_norm / total
    return total


def adamw_update(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay):
    # Written in "One training step and the loop around it".
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    theta = theta * (1 - lr * weight_decay) - lr * m_hat / (v_hat**0.5 + eps)
    return theta, m, v


DEFAULT_CONFIG = {
    "vocab": VOCAB,
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 64,
}


def init_params(cfg: dict, seed: int = 0) -> dict:
    """Given: seeded weights. Names and shapes define what `forward` reads."""
    g = torch.Generator().manual_seed(seed)
    d, f, V = cfg["d_model"], cfg["d_ff"], cfg["vocab"]

    def w(*shape, std):
        return (torch.randn(*shape, generator=g) * std).requires_grad_()

    def ones(n):
        return torch.ones(n, requires_grad=True)

    def zeros(n):
        return torch.zeros(n, requires_grad=True)

    params = {"embed": w(V, d, std=0.1), "ln_f_w": ones(d), "ln_f_b": zeros(d)}
    for i in range(cfg["n_layers"]):
        p = f"l{i}."
        for name in ("Wq", "Wk", "Wv", "Wo"):
            params[p + name] = w(d, d, std=d**-0.5)
        params[p + "ln1_w"], params[p + "ln1_b"] = ones(d), zeros(d)
        params[p + "ln2_w"], params[p + "ln2_b"] = ones(d), zeros(d)
        params[p + "Wg"] = w(d, f, std=d**-0.5)
        params[p + "Wu"] = w(d, f, std=d**-0.5)
        params[p + "Wd"] = w(f, d, std=f**-0.5)
    return params


# ---------------------------------------------------------------------------
# Your work.
# ---------------------------------------------------------------------------


def sinusoidal_positions(T: int, d: int) -> torch.Tensor:
    speeds = 10000.0 ** (-2.0 * torch.arange(d // 2, dtype=torch.float32) / d)
    angles = torch.arange(T, dtype=torch.float32)[:, None] * speeds[None, :]
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def causal_mask(T: int) -> torch.Tensor:
    mask = torch.zeros(T, T)
    mask[torch.triu(torch.ones(T, T), diagonal=1).bool()] = float("-inf")
    return mask


def causal_self_attention(x, Wq, Wk, Wv, Wo, n_heads: int) -> torch.Tensor:
    B, T, d = x.shape
    d_head = d // n_heads

    def heads(t):
        return t.view(B, T, n_heads, d_head).transpose(1, 2)  # (B, H, T, d_head)

    q, k, v = heads(x @ Wq), heads(x @ Wk), heads(x @ Wv)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_head) + causal_mask(T)
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ v).transpose(1, 2).reshape(B, T, d)
    return out @ Wo


def transformer_block(x, p: dict, prefix: str, n_heads: int) -> torch.Tensor:
    h = layer_norm(x, p[prefix + "ln1_w"], p[prefix + "ln1_b"])
    x = x + causal_self_attention(
        h, p[prefix + "Wq"], p[prefix + "Wk"], p[prefix + "Wv"], p[prefix + "Wo"], n_heads
    )
    h = layer_norm(x, p[prefix + "ln2_w"], p[prefix + "ln2_b"])
    gate = torch.nn.functional.silu(h @ p[prefix + "Wg"])
    return x + (gate * (h @ p[prefix + "Wu"])) @ p[prefix + "Wd"]


def forward(params: dict, ids: torch.Tensor, cfg: dict) -> torch.Tensor:
    B, T = ids.shape
    d = cfg["d_model"]
    x = params["embed"][ids] * math.sqrt(d) + sinusoidal_positions(T, d)
    for i in range(cfg["n_layers"]):
        x = transformer_block(x, params, f"l{i}.", cfg["n_heads"])
    x = layer_norm(x, params["ln_f_w"], params["ln_f_b"])
    return x @ params["embed"].T


def answer_loss(logits, targets, answer_mask) -> torch.Tensor:
    losses = per_token_loss(logits, targets)
    m = answer_mask.to(losses.dtype)
    return (losses * m).sum() / m.sum()


def parameter_count(cfg: dict) -> int:
    d, f, V, L = cfg["d_model"], cfg["d_ff"], cfg["vocab"], cfg["n_layers"]
    per_layer = 4 * d * d + 3 * d * f + 4 * d
    return V * d + L * per_layer + 2 * d


def training_flops(cfg: dict, n_tokens: int) -> float:
    return 6.0 * parameter_count(cfg) * n_tokens


def train(params: dict, data: torch.Tensor, cfg: dict, tcfg: dict):
    """Train in place of the given params; return (params, losses)."""
    names = list(params.keys())
    m = {k: torch.zeros_like(v) for k, v in params.items()}
    v = {k: torch.zeros_like(v) for k, v in params.items()}
    gen = torch.Generator().manual_seed(tcfg.get("seed", 0))
    no_decay = {k for k, t in params.items() if t.dim() < 2}
    losses = []
    answer_mask = torch.zeros(SEQ_LEN - 1, dtype=torch.bool)
    answer_mask[N_DIGITS:] = True
    for step in range(1, tcfg["steps"] + 1):
        idx = torch.randint(0, len(data), (tcfg["batch"],), generator=gen)
        batch = data[idx]
        logits = forward(params, batch[:, :-1], cfg)
        loss = answer_loss(logits, batch[:, 1:], answer_mask.expand(len(batch), -1))
        loss.backward()
        clip_global_norm([params[k].grad for k in names], tcfg["max_grad_norm"])
        lr = warmup_cosine_lr(step, tcfg["warmup"], tcfg["steps"], tcfg["peak_lr"], tcfg["floor_lr"])
        new = {}
        with torch.no_grad():
            for k in names:
                wd = 0.0 if k in no_decay else tcfg["weight_decay"]
                theta, m[k], v[k] = adamw_update(
                    params[k], params[k].grad, m[k], v[k], step, lr, 0.9, 0.95, 1e-8, wd
                )
                new[k] = theta.detach().requires_grad_()
        params = new
        losses.append(float(loss.item()))
    return params, losses


def greedy_decode(params: dict, prompts: torch.Tensor, n_new: int, cfg: dict) -> torch.Tensor:
    ids = prompts
    for _ in range(n_new):
        with torch.no_grad():
            logits = forward(params, ids, cfg)
        ids = torch.cat([ids, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    return ids


def exact_match(params: dict, data: torch.Tensor, cfg: dict) -> float:
    prompts = data[:, : N_DIGITS + 1]
    out = greedy_decode(params, prompts, N_DIGITS, cfg)
    return float((out == data).all(dim=1).float().mean())


DEFAULT_TRAIN = {
    "steps": 450, "batch": 64, "warmup": 40, "peak_lr": 3e-3, "floor_lr": 3e-4,
    "weight_decay": 0.01, "max_grad_norm": 1.0, "seed": 0,
}

if __name__ == "__main__":
    torch.manual_seed(0)
    data = make_data(0)
    cfg = DEFAULT_CONFIG
    params = init_params(cfg, seed=0)
    print("weights:", parameter_count(cfg), "of", sum(t.numel() for t in params.values()))
    tokens = DEFAULT_TRAIN["steps"] * DEFAULT_TRAIN["batch"] * (SEQ_LEN - 1)
    print(f"training flops: {training_flops(cfg, tokens):.2e}")
    params, losses = train(params, data["train"], cfg, DEFAULT_TRAIN)
    print(f"loss: first {losses[0]:.3f}, last {losses[-1]:.3f}")
    print("held-out exact match:", exact_match(params, data["test"], cfg))
    out = greedy_decode(params, data["test"][:3, : N_DIGITS + 1], N_DIGITS, cfg)
    for row in out:
        print("".join(">" if t == SEP else str(int(t)) for t in row))
