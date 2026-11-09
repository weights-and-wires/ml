"""Checks for the PyTorch bridge exercise.

`sgd_step_forgets_zero_grad` is the wrong step the doubled-update check is
aimed at. It subtracts the learning rate times the gradient and leaves the
gradient in place, so the next backward adds onto the old slope and the
second update doubles.
"""

import torch
from torch import nn

from mlcheck import almost, check, close, fail, has


def sgd_step_forgets_zero_grad(params, lr):
    """Wrong on purpose: do not zero the gradient.

    The second backward adds a fresh slope of 1 onto the leftover 1, so with
    learning rate 0.1 the second update is 0.2 instead of 0.1.
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad


def _reference_linear(in_features, out_features, seed):
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features)


def _load_linear(sub, ref):
    layer = has(sub, "Linear")(ref.in_features, ref.out_features)
    names = sorted(name for name, _ in layer.named_parameters())
    if names != ["bias", "weight"]:
        fail(f"registered parameters are {names}, expected ['bias', 'weight']")
    if tuple(layer.weight.shape) != tuple(ref.weight.shape):
        fail(
            f"weight shape is {tuple(layer.weight.shape)}, "
            f"expected {tuple(ref.weight.shape)}"
        )
    if tuple(layer.bias.shape) != tuple(ref.bias.shape):
        fail(
            f"bias shape is {tuple(layer.bias.shape)}, "
            f"expected {tuple(ref.bias.shape)}"
        )
    with torch.no_grad():
        layer.weight.copy_(ref.weight)
        layer.bias.copy_(ref.bias)
    return layer


def _two_updates(step_fn, lr=0.1):
    """Two steps on a scalar whose true slope is 1. Return the update sizes."""
    param = nn.Parameter(torch.tensor(1.0))
    sizes = []
    for _ in range(2):
        before = float(param.detach())
        param.sum().backward()
        step_fn([param], lr)
        sizes.append(before - float(param.detach()))
    return sizes, param


def _loop_attention(q, k, v):
    """The attention chapter's loop, repeated for every query in a batch."""
    batches, queries, _ = q.shape
    sources = k.shape[1]
    value_width = v.shape[-1]
    out = torch.empty(batches, queries, value_width, dtype=q.dtype)
    for batch in range(batches):
        for query in range(queries):
            scores = torch.stack(
                [torch.dot(q[batch, query], k[batch, source]) for source in range(sources)]
            )
            scores = scores - scores.max()
            weights = torch.exp(scores)
            weights = weights / weights.sum()
            context = torch.zeros(value_width, dtype=q.dtype)
            for source in range(sources):
                context = context + weights[source] * v[batch, source]
            out[batch, query] = context
    return out


def _reference_epoch(model, x, y, lr, batch_size, seed):
    """The same epoch as the lesson, with the gradient zeroed after each step."""
    model.train()
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(n, generator=generator)
    total = 0.0
    for start in range(0, n, batch_size):
        index = order[start : start + batch_size]
        pred = model(x[index])
        loss = ((pred - y[index]) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
                param.grad.zero_()
        total += float(loss.detach()) * index.shape[0]
    return total / n


@check("Linear matches torch.nn.Linear on a batch of two rows")
def test_linear_forward(sub):
    ref = _reference_linear(3, 2, seed=0)
    layer = _load_linear(sub, ref)
    x = torch.tensor([[1.0, 2.0, -1.0], [0.5, 0.0, 4.0]])
    close(layer(x).detach(), ref(x).detach(), atol=1e-6, label="Linear output")


@check("Linear gradients match torch.nn.Linear to 1e-6")
def test_linear_grad(sub):
    ref = _reference_linear(3, 2, seed=1)
    layer = _load_linear(sub, ref)
    x = torch.tensor([[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]])
    layer(x).sum().backward()
    ref(x).sum().backward()
    if layer.weight.grad is None or layer.bias.grad is None:
        fail("expected weight.grad and bias.grad after backward, got None")
    close(layer.weight.grad, ref.weight.grad, atol=1e-6, label="weight gradient")
    close(layer.bias.grad, ref.bias.grad, atol=1e-6, label="bias gradient")


@check("MLP matches a stack of torch.nn.Linear, including every registered gradient")
def test_mlp(sub):
    mlp_cls = has(sub, "MLP")
    torch.manual_seed(2)
    ref = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    mlp = mlp_cls([3, 4, 2])
    reader = list(mlp.parameters())
    official = [param for param in ref.parameters()]
    if len(reader) != len(official):
        fail(f"MLP registered {len(reader)} tensors, expected {len(official)}")
    count = sum(param.numel() for param in reader)
    if count != 26:
        fail(f"MLP([3, 4, 2]) registered {count} numbers, expected 26")
    with torch.no_grad():
        for got, want in zip(reader, official):
            if tuple(got.shape) != tuple(want.shape):
                fail(f"parameter shape is {tuple(got.shape)}, expected {tuple(want.shape)}")
            got.copy_(want)
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(5, 3, generator=generator)
    got = mlp(x)
    want = ref(x)
    close(got.detach(), want.detach(), atol=1e-6, label="MLP output")
    got.square().sum().backward()
    want.square().sum().backward()
    for index, (left, right) in enumerate(zip(mlp.parameters(), ref.parameters())):
        if left.grad is None:
            fail(f"parameter {index} gradient is None, expected a tensor")
        close(left.grad, right.grad, atol=1e-6, label=f"parameter {index} gradient")


@check("sgd_step subtracts lr times the gradient and leaves a zero gradient")
def test_sgd_step(sub):
    step = has(sub, "sgd_step")
    param = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    param.grad = torch.tensor([0.5, -0.25, 1.0])
    step([param], 0.1)
    close(param.detach(), torch.tensor([0.95, -1.975, 0.4]), atol=1e-6, label="updated parameter")
    if param.grad is not None:
        close(param.grad, torch.zeros(3), atol=0.0, label="gradient after the step")


@check("the second SGD update stays 0.1 when the true slope is 1")
def test_second_step_is_not_doubled(sub):
    # sgd_step_forgets_zero_grad leaves the slope in place. Its second update
    # is 0.2. A correct sgd_step zeros the slope, so the second update stays 0.1.
    step = has(sub, "sgd_step")
    got, param = _two_updates(step)
    bad, _ = _two_updates(sgd_step_forgets_zero_grad)
    if abs(bad[1] - 0.2) > 1e-5:
        fail(f"sgd_step_forgets_zero_grad second update is {bad[1]:.6g}, expected 0.2")
    if abs(got[1] - bad[1]) <= 1e-5:
        fail(
            f"second update is {got[1]:.6g}, expected 0.1. "
            f"sgd_step_forgets_zero_grad produces {bad[1]:.6g} because the leftover gradient doubles the step."
        )
    if abs(got[0] - 0.1) > 1e-5 or abs(got[1] - 0.1) > 1e-5:
        fail(f"updates are {got[0]:.6g} and {got[1]:.6g}, expected 0.1 and 0.1")
    leftover = 0.0 if param.grad is None else float(param.grad.detach().abs().max())
    if leftover != 0.0:
        fail(f"gradient after the second step is {leftover:.6g}, expected 0")


@check("train_epoch lowers the loss, matches a zeroing step, and leaves gradients at 0")
def test_train_epoch(sub):
    train_epoch = has(sub, "train_epoch")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    try:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(10, 1, generator=generator)
        y = 1.5 * x - 0.5
        reader = nn.Linear(1, 1)
        official = nn.Linear(1, 1)
        with torch.no_grad():
            reader.weight.zero_()
            reader.bias.zero_()
            official.weight.zero_()
            official.bias.zero_()
        initial = float(((reader(x) - y) ** 2).mean().detach())
        got = train_epoch(reader, x, y, 0.2, 4, 1)
        want = _reference_epoch(official, x, y, 0.2, 4, 1)
        almost(got, want, 1e-5, label="epoch loss")
        close(reader.weight.detach(), official.weight.detach(), atol=1e-5, label="weight after one epoch")
        close(reader.bias.detach(), official.bias.detach(), atol=1e-5, label="bias after one epoch")
        for _ in range(7):
            train_epoch(reader, x, y, 0.2, 4, 1)
        final = float(((reader(x) - y) ** 2).mean().detach())
        if not final < initial * 0.5:
            fail(
                f"full-batch loss is {final:.6g}, expected it below {0.5 * initial:.6g} "
                f"(the loss started at {initial:.6g})"
            )
        for name, param in reader.named_parameters():
            leftover = 0.0 if param.grad is None else float(param.grad.detach().abs().max())
            if leftover != 0.0:
                fail(f"{name} gradient max is {leftover:.6g}, expected 0")
    finally:
        torch.use_deterministic_algorithms(False)


@check("eval_mse returns 21 on the four points and leaves the module in eval mode")
def test_eval_mse(sub):
    eval_mse = has(sub, "eval_mse")
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    model.train()
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    got = eval_mse(model, x, y)
    almost(got, 21.0, 1e-5, label="eval mse")
    if model.training:
        fail("model.training is True, expected False")
    if not torch.is_grad_enabled():
        fail("gradient tracking is disabled after eval_mse, expected it enabled")


@check("einsum_attention matches the attention loop on the worked query")
def test_einsum_worked(sub):
    fn = has(sub, "einsum_attention")
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    close(fn(q, k, v), _loop_attention(q, k, v), atol=1e-6, label="worked context")


@check("einsum_attention matches the loop on a seeded batch and on a score of 800")
def test_einsum_batch_and_large_score(sub):
    fn = has(sub, "einsum_attention")
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(2, 3, 4, generator=generator)
    k = torch.randn(2, 5, 4, generator=generator)
    v = torch.randn(2, 5, 6, generator=generator)
    close(fn(q, k, v), _loop_attention(q, k, v), atol=1e-5, label="batched context")

    q = torch.tensor([[[1.0]]])
    k = torch.tensor([[[0.0], [800.0]]])
    v = torch.tensor([[[5.0, 1.0], [0.0, 4.0]]])
    got = fn(q, k, v)
    want = _loop_attention(q, k, v)
    close(got, want, atol=1e-5, label="context at a score of 800")
    close(want, torch.tensor([[[0.0, 4.0]]]), atol=1e-5, label="loop context at a score of 800")
