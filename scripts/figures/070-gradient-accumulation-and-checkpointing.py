#!/usr/bin/env python3
"""Numbers and SVG coordinates for the gradient accumulation and checkpointing lesson.

Figure 1 stacks the memory of one training step for a 24-layer, width-1024
model at three batch sizes. Figure 2 plots the checkpointing peak against the
segment count for 16 layers. The script also prints the worked hand examples.
"""

from __future__ import annotations

import math

import numpy as np

GB = 1e9


def params_of(layers, width):
    return 12 * layers * width * width


def activation_bytes(batch, width, layers, bytes_per_number=4):
    return batch * width * layers * bytes_per_number


def stored_peak(layers, segments):
    return segments + math.ceil(layers / segments) - 1


def main():
    layers, width, seq = 24, 1024, 1024
    n = params_of(layers, width)
    fixed = 16 * n
    print(f"params N = {n:,}; 16N bytes = {fixed / GB:.2f} GB")
    for batch in (8, 32, 128):
        act = activation_bytes(batch * seq, width, layers)
        print(f"batch {batch:4d}: tokens {batch * seq:7d}, one array per layer = {act / GB:.2f} GB, "
              f"total {(fixed + act) / GB:.2f} GB")
        # Figure 1 bar geometry: 14 GB spans 400 units.
        scale = 400 / 14
        print(f"   fixed bar {fixed / GB * scale:.0f} units, activation bar {act / GB * scale:.0f} units")

    print("\nHand example, one weight w = 1, loss 0.5 (w x - y)^2, examples (x, y):")
    ex = [(1, 3), (2, 2), (3, 0)]
    g = [(1 * x - y) * x for x, y in ex]
    print("per-example gradients", g, "mean", np.mean(g))
    a, b = np.mean(g[:1]), np.mean(g[1:])
    print("micro means", a, b, "weighted", a / 3 + b * 2 / 3, "average of means", (a + b) / 2, "sum", a + b)

    print("\nBatch norm example, values 1..6:")
    v = np.arange(1, 7, dtype=float)
    print("full mean", v.mean(), "var", v.var(), "| halves means", v[:3].mean(), v[3:].mean(), "var", v[:3].var())

    print("\nCheckpointing, 16 layers, peak by segment count:")
    peaks = [stored_peak(16, k) for k in range(1, 17)]
    print(peaks)
    ymax, base = 16, 190.0
    for k, p in enumerate(peaks, start=1):
        x = 70 + (k - 1) * 28
        h = p / ymax * 150
        print(f"k={k:2d} peak={p:2d} x={x} bar height {h:.1f} top y={base - h:.1f}")
    print("L=100: best k", min(range(1, 101), key=lambda k: (stored_peak(100, k), k)),
          "peak", min(stored_peak(100, k) for k in range(1, 101)))


if __name__ == "__main__":
    main()
