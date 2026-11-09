"""Reference solution: a Linear, an MLP, a zeroing SGD step, and einsum attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    """An affine map, y = x @ weight.T + bias.

    Args:
        in_features: Width of each input row.
        out_features: Width of each output row.

    weight has shape (out_features, in_features) and bias has shape
    (out_features,). Both are nn.Parameter, so parameters() yields them.
    forward takes x of shape (N, in_features) and returns (N, out_features).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features) if self.in_features else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x @ weight.T + bias."""
        return F.linear(x, self.weight, self.bias)


class MLP(nn.Module):
    """A stack of Linear layers with a ReLU between them.

    Args:
        sizes: Widths from input to output, for example [3, 4, 2].

    The last layer stays linear. parameters() yields every weight and bias.
    forward takes (N, sizes[0]) and returns (N, sizes[-1]).
    """

    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            Linear(int(sizes[i]), int(sizes[i + 1])) for i in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each Linear, with ReLU after every layer but the last."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def sgd_step(params, lr: float) -> None:
    """Subtract lr * grad from each parameter, then zero the gradient.

    Args:
        params: Parameters, each a tensor that may carry .grad.
        lr: Learning rate, a positive float.

    The update runs inside torch.no_grad(). A parameter with grad None is
    left alone. After the call, every gradient that was used is 0.
    """
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                continue
            param -= lr * param.grad
            param.grad.zero_()


def train_epoch(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    batch_size: int,
    seed: int,
) -> float:
    """One epoch of mean squared error. Return the mean over rows.

    Args:
        model: An nn.Module. The call starts with model.train().
        x: Inputs, shape (n, in_features).
        y: Targets, shape (n, out_features).
        lr: Learning rate passed to sgd_step.
        batch_size: Rows per step. The last block may be shorter.
        seed: Seed for a local torch.Generator that shuffles the rows.

    Returns:
        The mean over the n rows of the squared error, as a Python float.
    """
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
        sgd_step(model.parameters(), lr)
        total += float(loss.detach()) * index.shape[0]
    return total / n


def eval_mse(model, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error on the whole set, with the tape off.

    Args:
        model: An nn.Module. The call switches it to eval mode and leaves it
            there.
        x: Inputs, shape (n, in_features).
        y: Targets, shape (n, out_features).

    Returns:
        The mean of the squared error, as a Python float.
    """
    model.eval()
    with torch.no_grad():
        pred = model(x)
        return float(((pred - y) ** 2).mean())


def einsum_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention blend: scores, softmax over sources, weighted values.

    Args:
        q: Queries, shape (B, T, D).
        k: Keys, shape (B, S, D).
        v: Values, shape (B, S, D_v).

    Returns:
        Context, shape (B, T, D_v).
    """
    scores = torch.einsum("btd,bsd->bts", q, k)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bts,bsv->btv", weights, v)


if __name__ == "__main__":
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    layer = Linear(1, 1)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    print(f"eval at a zero line: {eval_mse(layer, x, y):.1f}")
    loss = train_epoch(layer, x, y, lr=0.05, batch_size=4, seed=0)
    print(f"loss after one full-batch step: {loss:.4f}")

    query = torch.tensor([[[1.0, 0.0]]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    values = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
    print(einsum_attention(query, keys, values).detach())
