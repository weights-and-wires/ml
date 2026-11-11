"""Project: train a tiny decoder-only transformer to reverse digit strings.

Sequences look like "483920>029384": six digits, a separator, then the same
six digits backwards. Everything above "Your work" is given: the data, and
helpers you wrote in earlier chapters. Fill in each function below it, in order.
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
    """Sinusoidal position table, as in "Where is each token?".

    Args:
        T: Number of positions.
        d: Width; even.

    Returns:
        A float32 tensor of shape (T, d). Column 2i holds sin(t * w_i) and column
        2i + 1 holds cos(t * w_i), where w_i = 10000 ** (-2 * i / d).

    Steps:
        1. Build the speeds w_i for i in range(d // 2).
        2. Multiply positions (T, 1) by speeds (1, d // 2) to get angles.
        3. Write sin into the even columns and cos into the odd columns.
    """
    raise NotImplementedError


def causal_mask(T: int) -> torch.Tensor:
    """Additive mask that blocks attention to later positions.

    Returns:
        A float tensor of shape (T, T): 0 where column <= row, -inf where
        column > row. Add it to the scores before the softmax.
    """
    raise NotImplementedError


def causal_self_attention(x, Wq, Wk, Wv, Wo, n_heads: int) -> torch.Tensor:
    """Multi-head self-attention with scaled scores and the causal mask.

    Args:
        x: Tensor of shape (B, T, d).
        Wq, Wk, Wv, Wo: Tensors of shape (d, d).
        n_heads: Number of heads; divides d.

    Returns:
        A tensor of shape (B, T, d). Output row t depends only on x[:, :t + 1].

    Steps:
        1. Project: q, k, v = x @ Wq, x @ Wk, x @ Wv.
        2. Split the width into heads: view (B, T, n_heads, d_head), then
           transpose to (B, n_heads, T, d_head).
        3. scores = q @ k^T / sqrt(d_head), then add causal_mask(T).
        4. Softmax over the last axis, then weights @ v.
        5. Put the heads back side by side to (B, T, d) and multiply by Wo.
    """
    raise NotImplementedError


def transformer_block(x, p: dict, prefix: str, n_heads: int) -> torch.Tensor:
    """One pre-norm block with a SwiGLU feed-forward.

    Args:
        x: Tensor of shape (B, T, d).
        p: The params dict. This block's weights are p[prefix + name] for
            ln1_w, ln1_b, Wq, Wk, Wv, Wo, ln2_w, ln2_b, Wg, Wu, Wd.
        prefix: For example "l0.".
        n_heads: Number of heads.

    Returns:
        A tensor of shape (B, T, d).

    Steps:
        1. x = x + attention(layer_norm(x, ln1)).
        2. h = layer_norm(x, ln2); the feed-forward is
           (silu(h @ Wg) * (h @ Wu)) @ Wd, with no biases.
        3. x = x + feed-forward. Norm the branch, never the residual stream.
    """
    raise NotImplementedError


def forward(params: dict, ids: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Token ids to next-token logits.

    Args:
        params: Weights from init_params. params["embed"] has shape (vocab, d)
            and is also the output matrix (weight tying).
        ids: Long tensor of shape (B, T).
        cfg: Dict with d_model, n_heads, n_layers.

    Returns:
        Logits of shape (B, T, vocab).

    Steps:
        1. x = params["embed"][ids] * sqrt(d_model) + sinusoidal_positions(T, d_model).
        2. Run n_layers blocks with prefixes "l0.", "l1.", and so on.
        3. Apply the final layer_norm with ln_f_w and ln_f_b.
        4. Multiply by params["embed"].T.
    """
    raise NotImplementedError


def answer_loss(logits, targets, answer_mask) -> torch.Tensor:
    """Mean next-token loss over the positions where answer_mask is True.

    Args:
        logits: (B, T, vocab).
        targets: Long tensor (B, T).
        answer_mask: Bool tensor (B, T). True where the target is an answer token.

    Returns:
        A scalar: the sum of per_token_loss over True positions divided by the
        number of True positions. Positions where the mask is False must
        contribute nothing, including nothing to the gradient.
    """
    raise NotImplementedError


def parameter_count(cfg: dict) -> int:
    """Number of weights in the model that init_params builds.

    Args:
        cfg: Dict with vocab, d_model, d_ff, n_layers.

    Returns:
        The total count, from a formula. Count the tied embedding once, four
        d x d attention matrices, three SwiGLU matrices, and every layer-norm
        gain and bias, including the final one. Do not add up init_params.
    """
    raise NotImplementedError


def training_flops(cfg: dict, n_tokens: int) -> float:
    """The 6ND estimate of a training run's flops, using parameter_count(cfg)."""
    raise NotImplementedError


def train(params: dict, data: torch.Tensor, cfg: dict, tcfg: dict):
    """Train the model. Return (params, losses).

    Args:
        params: Weights from init_params.
        data: Long tensor (n, 13) of full sequences.
        cfg: Model config.
        tcfg: Dict with steps, batch, warmup, peak_lr, floor_lr, weight_decay,
            max_grad_norm, seed.

    Returns:
        The trained params dict (fresh leaf tensors with requires_grad) and a
        list of the loss at every step.

    Steps, once per step number 1..steps:
        1. Draw `batch` row indices with a torch.Generator seeded by tcfg["seed"].
        2. Inputs are rows[:, :-1], targets are rows[:, 1:]. The answer mask is
           True for target positions N_DIGITS onward (the answer digits).
        3. Forward, answer_loss, loss.backward().
        4. clip_global_norm over every gradient, then lr from warmup_cosine_lr.
        5. adamw_update on each weight inside torch.no_grad(); no weight decay
           for tensors with fewer than two dimensions. Store
           theta.detach().requires_grad_().
    """
    raise NotImplementedError


def greedy_decode(params: dict, prompts: torch.Tensor, n_new: int, cfg: dict) -> torch.Tensor:
    """Extend each prompt by n_new tokens, always taking the argmax.

    Args:
        prompts: Long tensor (B, P).

    Returns:
        A long tensor (B, P + n_new) that starts with the prompts. Run the
        forward pass on everything so far, take the logits at the last
        position, append the argmax, and repeat.
    """
    raise NotImplementedError


def exact_match(params: dict, data: torch.Tensor, cfg: dict) -> float:
    """Fraction of rows in data that greedy decoding reproduces exactly.

    Give the model data[:, :N_DIGITS + 1] (the digits and the separator),
    decode N_DIGITS tokens, and count rows where all 13 tokens equal the row.
    """
    raise NotImplementedError


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
