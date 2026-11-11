#!/usr/bin/env python3
"""Numbers for the state-space models lesson.

Figure 1 compares the values read to produce one token: attention reads the
whole key-value cache, linear attention reads a fixed state.
Figure 2 draws the kernel K_j = a^j for a decay of 0.5, 0.9, and 1.05.
Figure 3 draws the rounds of a Hillis-Steele scan over 8 steps.

The script prints every value drawn or quoted in the lesson: the worked linear
attention example, the worked state-space example, the zero-order-hold pair,
the scan example, the memory arithmetic, and the scan rounds.
Run it with `python scripts/figures/066-state-space-models.py`.
"""

from __future__ import annotations

import math

import numpy as np


def main():
    print("== worked linear attention (phi values given)")
    pq = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pk = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = np.array([2.0, 4.0, 6.0])
    S = np.zeros((2,))
    z = np.zeros(2)
    for t in range(3):
        S = S + pk[t] * v[t]
        z = z + pk[t]
        print(f"t={t + 1} S={S} z={z} num={pq[t] @ S} den={pq[t] @ z} y={pq[t] @ S / (pq[t] @ z)}")
    scores = np.tril(pq @ pk.T)
    print("masked scores\n", scores, "\nrow sums", scores.sum(axis=1))

    print("== worked state-space system a=0.5, b=c=1, x=[1,2,3]")
    a, x = 0.5, [1.0, 2.0, 3.0]
    h, ys = 0.0, []
    for xt in x:
        h = a * h + xt
        ys.append(h)
    print("recurrent", ys, "kernel", [a**j for j in range(3)])
    print("convolution", [sum(a**j * x[t - j] for j in range(t + 1)) for t in range(3)])

    print("== zero-order hold, a=-1, b=1, dt=1")
    a_bar = math.exp(-1.0)
    print("a_bar", round(a_bar, 4), "b_bar", round((a_bar - 1) / -1.0, 4))
    print("a=-2 dt=0.5: a_bar", round(math.exp(-1.0), 4), "b_bar", round((math.exp(-1.0) - 1) / -2.0, 4))

    print("== Figure 2: kernel a^j, j=0..11")
    for a in (0.5, 0.9, 1.05):
        print(a, [round(a**j, 3) for j in range(12)])

    print("== scan example a=[0.5,2,3], b=[1,1,1]")
    h = 0.0
    for at, bt in [(0.5, 1.0), (2.0, 1.0), (3.0, 1.0)]:
        h = at * h + bt
        print(" h", h)
    print("combine (0.5,1),(2,1) ->", (0.5 * 2, 2 * 1 + 1), "wrong b1+b2 ->", 1 + 1)

    print("== Figure 3: Hillis-Steele over T=8")
    a = np.array([0.5, 0.9, 0.8, 0.7, 0.9, 0.6, 0.8, 0.5])
    b = np.ones(8)
    A, B = a.copy(), b.copy()
    shift, rnd = 1, 0
    print("round 0 b", B.round(3))
    while shift < 8:
        nA = np.concatenate([A[:shift], A[shift:] * A[:-shift]])
        nB = np.concatenate([B[:shift], A[shift:] * B[:-shift] + B[shift:]])
        A, B = nA, nB
        shift *= 2
        rnd += 1
        print(f"round {rnd} window {shift} b", B.round(3))
    h, seq = 0.0, []
    for at, bt in zip(a, b):
        h = at * h + bt
        seq.append(h)
    print("loop h", np.array(seq).round(3), "match", np.allclose(seq, B))
    print("depth for T=8:", math.ceil(math.log2(8)), " T=4096:", math.ceil(math.log2(4096)),
          " T=1,000,000:", math.ceil(math.log2(1_000_000)))
    print("work: loop", 8 - 1, "combines; Hillis-Steele", sum(8 - s for s in (1, 2, 4)))

    print("== Figure 1: values read per token, d = dv = 64")
    d = 64
    state = d * d + d
    print("linear attention state", state, "values")
    for t in (1, 16, 32, 33, 64, 128, 256, 4096):
        print(f"t={t} attention reads {2 * t * d}  state {state}  ratio {2 * t * d / state:.2f}")
    print("crossover t =", state / (2 * d))
    print("total reads for T=4096: attention", sum(2 * t * d for t in range(1, 4097)), " state", 4096 * state)


if __name__ == "__main__":
    main()
