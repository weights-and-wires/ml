"""Train a language model safely: next-token loss, perplexity, a schedule, and a clip.

You write six functions: `per_token_loss`, `perplexity`, `make_lm_batch`,
`warmup_cosine_lr`, `global_norm`, and `clip_global_norm`. The functions
`causal_lm_loss` and `lr_trace` are already written. The first averages your
per-token costs into the single number that training pushes down. The second
calls your schedule at evenly spaced steps so you can see its shape.

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


def perplexity(loss) -> float:
    """Read a mean loss as a number of equally likely choices: exp(loss).

    Args:
        loss: The mean loss in nats, a float or a one-element tensor.

    Returns:
        The perplexity, as a plain float.
    """
    raise NotImplementedError("perplexity")


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


def warmup_cosine_lr(step: float, warmup: float, total: float, peak: float, floor: float) -> float:
    """The learning rate at one step: a straight ramp, then half a cosine.

    Below `warmup`, the rate is peak * step / warmup. From `warmup` to
    `total` it glides from `peak` down to `floor` along a cosine, and it
    equals `floor` at step == total. The two pieces agree at step == warmup.

    The function maps any number to a rate. `train_step` will pass the count
    of the update about to be applied, starting at 1.

    Args:
        step: The step number. May be a float.
        warmup: The number of warmup steps.
        total: The number of steps in the schedule, total > warmup.
        peak: The rate at the end of warmup.
        floor: The rate at step == total.

    Returns:
        The learning rate, as a float.
    """
    raise NotImplementedError("warmup_cosine_lr")


def global_norm(grads: list) -> float:
    """The length of all the gradients taken together.

    Treat every entry of every array as one long vector, and return the
    square root of the sum of the squares of all entries. Do not change the
    arrays.

    Args:
        grads: A list of numpy arrays or torch tensors of any shapes.

    Returns:
        The global norm, as a plain float.
    """
    raise NotImplementedError("global_norm")


def clip_global_norm(grads: list, max_norm: float) -> float:
    """Shorten the gradients in place so their global norm is at most `max_norm`.

    When the global norm is over `max_norm`, multiply every array, in place,
    by the same number max_norm / norm. When it is not, change nothing.

    Args:
        grads: A list of numpy arrays or torch tensors of any shapes.
        max_norm: The limit on the global norm.

    Returns:
        The global norm from before any clipping, as a float.
    """
    raise NotImplementedError("clip_global_norm")



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
