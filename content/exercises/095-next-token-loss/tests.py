"""Checks for the next-token loss exercise."""

import math

import numpy as np
import torch

from mlcheck import almost, check, close, equal, fail, has


@check("per_token_loss matches minus the log probability on the worked example")
def test_worked_example(sub):
    got = has(sub, "per_token_loss")
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    out = got(logits, targets)
    equal(tuple(out.shape), (2,), label="shape of the per-token costs")
    close(out.detach().numpy(), np.array([0.2395, 0.0949]), atol=5e-5, label="per-token costs")


@check("per_token_loss stays finite and exact on logits of 1000")
def test_stability(sub):
    got = has(sub, "per_token_loss")
    logits = torch.tensor([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [1000.0, 999.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 1])
    out = got(logits, targets).detach().numpy()
    if not np.all(np.isfinite(out)):
        fail(f"per_token_loss returned non-finite values {out}; subtract the largest logit before exponentiating")
    want = np.array([0.0, 1000.0, 1.0 + math.log(1 + math.exp(-1))])
    close(out, want, atol=1e-8, label="costs at huge logits")


@check("per_token_loss handles any leading shape")
def test_shapes(sub):
    got = has(sub, "per_token_loss")
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(3, 5, 7)))
    targets = torch.tensor(rng.integers(0, 7, size=(3, 5)))
    out = got(logits, targets)
    equal(tuple(out.shape), (3, 5), label="shape for (B, T, V) logits")
    z = logits.numpy()
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    want = -np.log(np.take_along_axis(p, targets.numpy()[..., None], -1)[..., 0])
    close(out.detach().numpy(), want, atol=1e-10, label="costs for (B, T, V) logits")


@check("The gradient of the mean loss is (softmax - one-hot) / N, confirmed by finite differences")
def test_gradient(sub):
    got = has(sub, "causal_lm_loss")
    rng = np.random.default_rng(1)
    z = rng.normal(size=(4, 5))
    t = torch.tensor(rng.integers(0, 5, size=4))

    def loss_at(arr):
        return float(got(torch.tensor(arr), t))

    numeric = np.zeros_like(z)
    h = 1e-6
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            up, down = z.copy(), z.copy()
            up[i, j] += h
            down[i, j] -= h
            numeric[i, j] = (loss_at(up) - loss_at(down)) / (2 * h)

    leaf = torch.tensor(z, requires_grad=True)
    got(leaf, t).backward()
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    onehot = np.eye(5)[t.numpy()]
    close(leaf.grad.numpy(), numeric, atol=1e-6, label="autograd against finite differences")
    close(leaf.grad.numpy(), (p - onehot) / 4, atol=1e-8, label="autograd against (softmax - one-hot) / N")


@check("Uniform logits score log V, and the perplexity is V")
def test_uniform(sub):
    per = has(sub, "per_token_loss")
    ppl = has(sub, "perplexity")
    for V in (11, 50_000):
        logits = torch.zeros(4, V, dtype=torch.float64)
        targets = torch.tensor([0, 1, 2, 3])
        loss = float(per(logits, targets).mean())
        almost(loss, math.log(V), tol=1e-9, label=f"loss on uniform logits, V = {V}")
        almost(ppl(loss), V, tol=1e-6 * V, label=f"perplexity, V = {V}")


@check("perplexity reads the loss as a count of choices")
def test_perplexity(sub):
    ppl = has(sub, "perplexity")
    almost(ppl(0.0), 1.0, tol=1e-12, label="a loss of 0 is one choice")
    almost(ppl(math.log(4.0)), 4.0, tol=1e-9, label="a loss of ln 4 is four choices")
    almost(ppl(torch.tensor(0.16723386)), 1.18203, tol=1e-4, label="the worked loss 0.1672")
    if not isinstance(ppl(0.5), float):
        fail(f"perplexity should return a plain float, got {type(ppl(0.5)).__name__}")


@check("make_lm_batch shifts the targets one place left")
def test_batch(sub):
    got = has(sub, "make_lm_batch")
    tokens = torch.arange(20) * 3
    inputs, targets = got(tokens, 5, 3)
    close(np.asarray(inputs), np.array([9, 12, 15, 18, 21]), label="inputs")
    close(np.asarray(targets), np.array([12, 15, 18, 21, 24]), label="targets")
    inputs, targets = got(tokens, 19, 0)
    equal(len(inputs), 19, label="the longest window")
    close(np.asarray(targets)[-1], 57, label="the last target is the last token")


@check("make_lm_batch refuses a window that runs past the end")
def test_batch_edge(sub):
    got = has(sub, "make_lm_batch")
    tokens = torch.arange(10)
    try:
        got(tokens, 5, 5)
    except ValueError:
        return
    except Exception as err:  # noqa: BLE001
        fail(f"expected ValueError, got {type(err).__name__}: {err}")
    fail("start 5 with block_size 5 needs 11 tokens but there are 10; expected a ValueError")


@check("A bigram table scores a teacher-forced window the way a hand loop does")
def test_bigram(sub):
    batch = has(sub, "make_lm_batch")
    per = has(sub, "per_token_loss")
    rng = np.random.default_rng(2)
    V = 6
    table = rng.normal(size=(V, V))
    tokens = torch.tensor(rng.integers(0, V, size=30))
    inputs, targets = batch(tokens, 12, 4)
    got = per(torch.tensor(table)[inputs], targets).detach().numpy()
    want = []
    for i in range(12):
        row = table[int(tokens[4 + i])]
        want.append(math.log(np.exp(row).sum()) - row[int(tokens[5 + i])])
    close(got, np.array(want), atol=1e-10, label="costs along a window")
