"""Reference solution for the next-token loss exercise."""

import math

import torch


def per_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def make_lm_batch(tokens, block_size: int, start: int):
    if start < 0 or start + block_size + 1 > len(tokens):
        raise ValueError(
            f"a window of {block_size} lessons starting at {start} needs "
            f"{start + block_size + 1} tokens, but there are {len(tokens)}"
        )
    inputs = tokens[start : start + block_size]
    targets = tokens[start + 1 : start + block_size + 1]
    return inputs, targets


def perplexity(loss) -> float:
    return math.exp(float(loss))


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Given: the mean of `per_token_loss`, one scalar to push down."""
    return per_token_loss(logits, targets).mean()


if __name__ == "__main__":
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
