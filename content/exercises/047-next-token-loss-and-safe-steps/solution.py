"""Reference solution: next-token loss, perplexity, shifted windows, a schedule, and a clip."""

import math

import torch


def per_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def perplexity(loss) -> float:
    return math.exp(float(loss))


def make_lm_batch(tokens, block_size: int, start: int):
    if start < 0 or start + block_size + 1 > len(tokens):
        raise ValueError(
            f"a window of {block_size} lessons starting at {start} needs "
            f"{start + block_size + 1} tokens, but there are {len(tokens)}"
        )
    inputs = tokens[start : start + block_size]
    targets = tokens[start + 1 : start + block_size + 1]
    return inputs, targets


def warmup_cosine_lr(step: float, warmup: float, total: float, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def global_norm(grads: list) -> float:
    return math.sqrt(sum(float((g**2).sum()) for g in grads))


def clip_global_norm(grads: list, max_norm: float) -> float:
    total = global_norm(grads)
    if total > max_norm:
        scale = max_norm / total
        for g in grads:
            g *= scale
    return total


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Given: the mean of `per_token_loss`, one scalar to push down."""
    return per_token_loss(logits, targets).mean()


def lr_trace(warmup: float, total: float, peak: float, floor: float, n_points: int) -> list:
    """Given: the schedule at `n_points` evenly spaced steps from 0 to `total`."""
    return [
        warmup_cosine_lr(total * i / (n_points - 1), warmup, total, peak, floor)
        for i in range(n_points)
    ]


if __name__ == "__main__":
    # The worked example: after "to" the model says "be", after "to be" it says "or".
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    targets = torch.tensor([0, 1])
    costs = per_token_loss(logits, targets)
    print("per-token costs:", [round(float(c), 4) for c in costs])
    print("mean loss:", round(float(causal_lm_loss(logits, targets)), 4))
    print("perplexity:", round(perplexity(causal_lm_loss(logits, targets)), 3))

    text = "to be or not to be, that is the question. " * 20
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    V = len(chars)
    inputs, shifted = make_lm_batch(data, block_size=8, start=0)
    print("inputs :", "".join(chars[int(i)] for i in inputs))
    print("targets:", "".join(chars[int(i)] for i in shifted))

    g = torch.Generator().manual_seed(0)
    table = (torch.randn(V, V, generator=g) * 0.02).requires_grad_()
    print(f"log V = {math.log(V):.3f}")
    for step in range(201):
        inputs, targets = make_lm_batch(data, block_size=len(data) - 1, start=0)
        loss = causal_lm_loss(table[inputs], targets)
        loss.backward()
        with torch.no_grad():
            table -= 1.0 * table.grad
            table.grad = None
        if step % 50 == 0:
            value = float(loss.detach())
            print(f"step {step:3d}  loss {value:.3f}  perplexity {perplexity(value):.2f}")

    trace = lr_trace(warmup=10, total=50, peak=2.0, floor=0.1, n_points=26)
    print("the schedule, one bar every two steps:")
    for i, value in enumerate(trace):
        print(f"step {2 * i:2d}  {value:5.3f}  {'#' * round(value * 20)}")

    grads = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    print("global norm:", global_norm(grads))
    before = clip_global_norm(grads, max_norm=2.5)
    print("norm before the clip:", before)
    print("after the clip:", [a.tolist() for a in grads])
