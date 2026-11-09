#!/usr/bin/env python3
"""Numbers for the from-the-engine-to-pytorch figure.

Figure 1 is the mean squared error of a one-weight line on four points,
y = 2x + 1, starting from a zero weight and a zero bias. Each step is
full-batch SGD with learning rate 0.05. There is no random draw: the four
points and the start are fixed.

The lesson SVG uses the polyline printed under "curve". Re-run this file
before changing that figure.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def fit_line(steps: int = 20, lr: float = 0.05):
    """Return the loss before each update, and the weight and bias after the last one."""
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    weight = nn.Parameter(torch.zeros(1, 1))
    bias = nn.Parameter(torch.zeros(1))
    losses = []
    for step in range(steps + 1):
        pred = x @ weight + bias
        loss = ((pred - y) ** 2).mean()
        losses.append(float(loss.detach()))
        if step == steps:
            break
        loss.backward()
        with torch.no_grad():
            weight -= lr * weight.grad
            bias -= lr * bias.grad
            weight.grad.zero_()
            bias.grad.zero_()
    return losses, float(weight.detach()), float(bias.detach())


def polyline(losses: list[float]) -> str:
    """Map step and log10(loss) into the lesson's 600 by 280 viewBox."""
    left, right, top, bottom = 72.0, 576.0, 36.0, 228.0
    log_max = math.log10(30.0)
    log_min = math.log10(2e-4)
    last = len(losses) - 1
    points = []
    for step, loss in enumerate(losses):
        px = left + (right - left) * step / last
        py = top + (log_max - math.log10(loss)) / (log_max - log_min) * (bottom - top)
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def grid_y(loss: float) -> float:
    """Vertical position of a gridline, using the same scale as `polyline`."""
    top, bottom = 36.0, 228.0
    log_max = math.log10(30.0)
    log_min = math.log10(2e-4)
    return top + (log_max - math.log10(loss)) / (log_max - log_min) * (bottom - top)


def main() -> None:
    losses, weight, bias = fit_line()
    print("loss before any step", f"{losses[0]:.6f}")
    print("loss after 1 step", f"{losses[1]:.6f}")
    print("loss after 2 steps", f"{losses[2]:.6f}")
    print("loss after 20 steps", f"{losses[-1]:.6f}")
    print("weight after 20 steps", f"{weight:.6f}")
    print("bias after 20 steps", f"{bias:.6f}")
    print("grid")
    for mark in (10.0, 1.0, 0.1, 0.01, 0.001):
        print(f"  {mark:g} {grid_y(mark):.1f}")
    print("curve")
    print(polyline(losses))


if __name__ == "__main__":
    main()
