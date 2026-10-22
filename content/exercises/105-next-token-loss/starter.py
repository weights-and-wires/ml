"""Score a language model: per-token loss, shifted windows, and perplexity.

You write three functions: `per_token_loss`, `make_lm_batch`, and
`perplexity`. The function `causal_lm_loss` is already written. It averages
your per-token costs into the single number that training pushes down.

The `__main__` block trains a bigram table, a model that guesses the next
character from the current one only, so you can watch the loss fall from
log V, the score of a model that knows nothing.
"""

import math

import torch


def per_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cost of every position: minus the log probability of the true token.

    Compute log probabilities as `logits - logsumexp(logits)` along the last
    axis, so large logits never overflow, then pick the entry at each target.

    Args:
        logits: Scores, shape (..., V).
        targets: Integer token ids, shape logits.shape[:-1].

    Returns:
        The unreduced costs, shape targets.shape, differentiable with
        respect to logits.
    """
    raise NotImplementedError("per_token_loss")


def make_lm_batch(tokens, block_size: int, start: int):
    """Cut one training window out of a token sequence.

    The inputs are `block_size` tokens beginning at `start`. The targets are
    the same window moved one place to the left, so targets[i] is the token
    that follows inputs[i].

    Args:
        tokens: A 1-D sequence of token ids (a torch tensor or a numpy array).
        block_size: The number of lessons in the window.
        start: Where the window begins.

    Returns:
        A pair (inputs, targets), each of length block_size.

    Raises:
        ValueError: If the window plus its one extra target token would run
            past the end of `tokens`, or if `start` is negative.
    """
    raise NotImplementedError("make_lm_batch")


def perplexity(loss) -> float:
    """Read a mean loss as a number of equally likely choices: exp(loss).

    Args:
        loss: The mean loss in nats, a float or a one-element tensor.

    Returns:
        The perplexity, as a plain float.
    """
    raise NotImplementedError("perplexity")


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Given: the mean of `per_token_loss`, one scalar to push down."""
    return per_token_loss(logits, targets).mean()


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

    # A bigram model: one row of logits per current character.
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
