#!/usr/bin/env python3
"""Numbers for the online softmax and tiled attention lesson.

Figure 1 draws the streaming softmax over the scores [2, 1, 5, 3]: the running
max m and running sum l after each score, with the rescale at the third score.
Figure 2 draws the tile grid of a causal table with T = 8 and block = 2.

The script prints every value drawn or quoted in the lesson: the worked
streaming softmax, the worked two-block attention (with and without the
rescale), the tile counts, and the memory and flop arithmetic for T = 4096.
Run it with `python scripts/figures/049-online-softmax-and-tiled-attention.py`.
"""

from __future__ import annotations

import numpy as np

SCORES = [2.0, 1.0, 5.0, 3.0]
VALUES = [10.0, 20.0, 30.0, 40.0]


def streaming(scores):
    m, l = -np.inf, 0.0
    rows = []
    for x in scores:
        m_new = max(m, x)
        scale = 0.0 if m == -np.inf else float(np.exp(m - m_new))
        l = l * scale + float(np.exp(x - m_new))
        m = m_new
        rows.append((x, m, scale, l))
    return rows


def block_state(s, v):
    m = max(s)
    e = [float(np.exp(x - m)) for x in s]
    return m, sum(e), sum(a * b for a, b in zip(e, v))


def main():
    print("== streaming softmax of", SCORES)
    rows = streaming(SCORES)
    for x, m, scale, l in rows:
        print(f"x={x:.0f} m={m:.0f} rescale={scale:.4f} l={l:.4f}")
    m, l = rows[-1][1], rows[-1][3]
    probs = [float(np.exp(x - m) / l) for x in SCORES]
    print("probs", [round(p, 4) for p in probs], "sum", round(sum(probs), 12))

    print("== two blocks, values", VALUES)
    a = block_state(SCORES[:2], VALUES[:2])
    b = block_state(SCORES[2:], VALUES[2:])
    print("block 1 (m, l, o)", [round(t, 4) for t in a])
    print("block 2 (m, l, o)", [round(t, 4) for t in b])
    mm = max(a[0], b[0])
    alpha, beta = np.exp(a[0] - mm), np.exp(b[0] - mm)
    l_new = a[1] * alpha + b[1] * beta
    o_new = a[2] * alpha + b[2] * beta
    print(f"alpha={alpha:.4f} beta={beta:.4f} l={l_new:.4f} o={o_new:.4f} out={o_new / l_new:.4f}")
    dense = sum(p * v for p, v in zip(probs, VALUES))
    wrong = (a[2] + b[2]) / l_new
    print(f"dense out={dense:.4f}  without rescaling o: o={a[2] + b[2]:.4f} out={wrong:.4f}")

    print("== tile grid T=8 block=2 causal")
    nb = 4
    on = nb
    below = nb * (nb - 1) // 2
    above = nb * (nb - 1) // 2
    print(f"diagonal={on} full below={below} skipped above={above} computed={on + below} of {nb * nb}")
    nb = 32
    print("T=4096 block=128: tiles", nb * nb, "computed", nb * (nb + 1) // 2, "fraction", round(nb * (nb + 1) / 2 / nb**2, 4))

    print("== memory, T=4096, 16 heads, float32")
    T, h, d, B = 4096, 16, 1024, 128
    table = T * T
    print("one table per head", table, "all heads", h * table, "bytes", 4 * h * table)
    print("dense saved S and P, all heads", 2 * h * table, "bytes", 4 * 2 * h * table, "GiB", round(4 * 2 * h * table / 2**30, 3))
    print("tiled saved lse, all heads", h * T, "bytes", 4 * h * T)
    print("tile in flight per head", B * B, "bytes", 4 * B * B)

    print("== flops, T=4096, d=1024 (all heads)")
    print("forward 4Td", 4 * T * T * d, "backward stored P 8T2d", 8 * T * T * d, "backward recompute 10T2d", 10 * T * T * d)
    print("training step total stored", 12 * T * T * d, "recompute", 14 * T * T * d, "ratio", round(14 / 12, 4))
    dh = d // h
    print("score traffic dense 4T2", 4 * T * T, "tiled K,V re-reads 2 T2 dh / B", 2 * T * T * dh // B, "ratio", 4 * T * T / (2 * T * T * dh / B))


if __name__ == "__main__":
    main()
