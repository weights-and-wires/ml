"""Map a handwritten tensor onto PyTorch and train one small module.

Fill in Linear, MLP, sgd_step, train_epoch, eval_mse, and einsum_attention.
Linear and MLP register their parameters. sgd_step subtracts and then zeros
the gradient. train_epoch and eval_mse are the two loops. einsum_attention
is the attention blend, scores then softmax then a weighted sum, written
with einsum.
"""

from __future__ import annotations

import torch
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
        raise NotImplementedError("Linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x @ weight.T + bias."""
        raise NotImplementedError("Linear.forward")


class MLP(nn.Module):
    """A stack of Linear layers with a ReLU between them.

    Args:
        sizes: Widths from input to output, for example [3, 4, 2].

    The last layer stays linear. parameters() yields every weight and bias.
    forward takes (N, sizes[0]) and returns (N, sizes[-1]).
    """

    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        raise NotImplementedError("MLP")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each Linear, with ReLU after every layer but the last."""
        raise NotImplementedError("MLP.forward")


def sgd_step(params, lr: float) -> None:
    """Subtract lr * grad from each parameter, then zero the gradient.

    Args:
        params: Parameters, each a tensor that may carry .grad.
        lr: Learning rate, a positive float.

    The update runs inside torch.no_grad(). A parameter with grad None is
    left alone. After the call, every gradient that was used is 0 or None.
    """
    raise NotImplementedError("sgd_step")


def train_epoch(model, x: torch.Tensor, y: torch.Tensor, lr: float, batch_size: int, seed: int) -> float:
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
        Weight each batch mean by the number of rows in that batch.
    """
    raise NotImplementedError("train_epoch")


def eval_mse(model, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error on the whole set, with the tape off.

    Args:
        model: An nn.Module. The call switches it to eval mode and leaves it
            there.
        x: Inputs, shape (n, in_features).
        y: Targets, shape (n, out_features).

    Returns:
        The mean of the squared error, as a Python float. The forward pass
        runs inside torch.no_grad().
    """
    raise NotImplementedError("eval_mse")


def einsum_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention blend: scores, softmax over sources, weighted values.

    Args:
        q: Queries, shape (B, T, D).
        k: Keys, shape (B, S, D).
        v: Values, shape (B, S, D_v).

    Returns:
        Context, shape (B, T, D_v). Softmax runs over the source axis, and
        the scores are shifted by their max so a score of 800 stays finite.
    """
    raise NotImplementedError("einsum_attention")


if __name__ == "__main__":
    layer = Linear(1, 1)
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    print(layer(x))
