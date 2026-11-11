#!/usr/bin/env python3
"""Numbers for the encoders and cross-attention lesson.

Figure 1 draws three attention masks for a 5-token sequence whose last token
is padding: encoder self-attention, decoder self-attention, and cross-attention
from 3 decoder positions to the 5 encoder positions.
Figure 2 draws how many positions give a training signal per 512-token
sequence under three objectives.

The script prints every value drawn or quoted in the lesson. Pass --svg to
print the SVG for both figures instead. Run it with
`python scripts/figures/065-encoders-and-cross-attention.py`.
"""

from __future__ import annotations

import sys

import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def masks(T=5, real=4):
    pad = np.arange(T) < real
    bidir = np.tile(pad, (T, 1))
    causal = np.tril(np.ones((T, T), dtype=bool)) & pad[None, :]
    cross = np.tile(pad, (3, 1))
    return pad, bidir, causal, cross


def span_lengths(T, density=0.15, mean_span=3.0):
    noise = int(round(T * density))
    spans = int(round(noise / mean_span))
    return T - noise + spans, noise + spans + 1, spans, noise


def values():
    print("--- masks (T=5, real length 4) ---")
    pad, bidir, causal, cross = masks()
    print("bidirectional, row 0:", bidir[0].astype(int), " causal row 0:", causal[0].astype(int))
    print("causal row 3:", causal[3].astype(int), " causal row 4 (a pad query):", causal[4].astype(int))
    print("allowed counts: bidir", int(bidir.sum()), "causal", int(causal.sum()), "cross", int(cross.sum()))

    print("--- cross-attention worked example ---")
    q = np.array([1.0, 0.0])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    scores = K @ q / np.sqrt(2)
    print("scores / sqrt(d):", np.round(scores, 4))
    w_all = softmax(scores)
    print("weights with all 3 keys:", np.round(w_all, 4), "output", np.round(w_all @ V, 4))
    w = softmax(scores[:2])
    print("weights with key 3 padded:", np.round(w, 4), "output", np.round(w @ V[:2], 4))
    print("exp(0.7071), exp(0):", round(float(np.exp(scores[0])), 4), 1.0)

    print("--- masked-LM loss worked example ---")
    logits = np.array([[2.0, 0.5, -1.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]])
    targets = np.array([0, 0, 2])
    chosen = np.array([True, False, True])
    p = np.array([softmax(r) for r in logits])
    nll = -np.log(p[np.arange(3), targets])
    print("probabilities of the target:", np.round(p[np.arange(3), targets], 4))
    print("per-position loss:", np.round(nll, 4))
    print("loss on chosen only:", round(float(nll[chosen].mean()), 4))
    print("loss over all positions:", round(float(nll.mean()), 4))
    g = p.copy()
    g[np.arange(3), targets] -= 1.0
    g = g * chosen[:, None] / chosen.sum()
    print("gradient rows:", np.round(g, 4).tolist())

    print("--- span corruption worked example ---")
    words = "Thank you for inviting me to your party last week".split()
    spans = [(2, 2), (8, 1)]
    inp, tgt, prev = [], [], 0
    for k, (s, n) in enumerate(spans):
        inp += words[prev:s] + [f"<{'XYZ'[k]}>"]
        tgt += [f"<{'XYZ'[k]}>"] + words[s:s + n]
        prev = s + n
    inp += words[prev:]
    tgt += [f"<{'XYZ'[len(spans)]}>"]
    print("input :", " ".join(inp))
    print("target:", " ".join(tgt))

    print("--- training signal per 512-token sequence ---")
    T = 512
    causal = T - 1
    mlm = int(round(0.15 * T))
    inl, tl, ns, nn = span_lengths(T)
    print("causal LM targets:", causal, " MLM chosen:", mlm)
    print("span corruption: noise tokens", nn, "spans", ns, "input length", inl, "target length", tl)
    print("share of the 512 tokens that give signal: causal", round(causal / T, 3),
          "mlm", round(mlm / T, 3), "span", round(nn / T, 3), "(target tokens", round(tl / T, 3), ")")
    print("span_lengths(100):", span_lengths(100))

    print("--- 80/10/10 expectation on 1000 chosen tokens ---")
    print("mask token 800, random token 100, kept 100")


def svg():
    pad, bidir, causal, cross = masks()
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Three attention masks over five positions where the last position is padding: encoder self-attention allows every real key, decoder self-attention allows only keys at or before the query, and cross-attention lets three decoder queries read every real encoder key">']
    c = 22
    panels = [
        ("encoder self-attention", bidir, 20, "queries: encoder", "keys: encoder"),
        ("decoder self-attention", causal, 230, "queries: decoder", "keys: decoder"),
        ("cross-attention", cross, 440, "queries: decoder", "keys: encoder"),
    ]
    for title, m, x0, _, _ in panels:
        out.append(f'<text x="{x0 + 55}" y="20" text-anchor="middle" class="fig-bold fig-small">{title}</text>')
        y0 = 40
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                cls = "fill-a" if m[i, j] else "fill-sunk"
                out.append(f'<rect x="{x0 + j * c}" y="{y0 + i * c}" width="{c - 2}" height="{c - 2}" class="{cls}"/>')
        out.append(f'<text x="{x0 + 4 * c + 9}" y="{y0 + m.shape[0] * c + 14}" text-anchor="middle" class="fig-small">pad</text>')
        out.append(f'<text x="{x0 + 55}" y="{y0 + 5 * c + 32}" text-anchor="middle" class="fig-small">columns are keys</text>')
    out.append('<text x="300" y="238" text-anchor="middle" class="fig-small">Blue: attention allowed. Grey: blocked. Rows are queries.</text>')
    out.append('</svg>')
    print("\n".join(out))
    print("=====")
    T = 512
    vals = [("causal LM", T - 1), ("masked LM", int(round(0.15 * T))), ("span corruption", span_lengths(T)[1])]
    out = ['<svg viewBox="0 0 600 190" role="img" aria-label="Bars for the number of target positions per 512-token sequence: causal language model 511, masked language model 77, span corruption 104">']
    for k, (name, n) in enumerate(vals):
        y = 24 + k * 48
        w = n / T * 340
        cls = ["fill-a", "fill-b", "fill-c"][k]
        out.append(f'<text x="140" y="{y + 18}" text-anchor="end" class="fig-bold fig-small">{name}</text>')
        out.append(f'<rect x="150" y="{y}" width="{w:.1f}" height="26" rx="3" class="{cls}"/>')
        out.append(f'<text x="{150 + w + 8:.1f}" y="{y + 18}" class="fig-small">{n} targets</text>')
    out.append('<text x="300" y="178" text-anchor="middle" class="fig-small">Targets the loss is computed on, per sequence of 512 tokens.</text>')
    out.append('</svg>')
    print("\n".join(out))


if __name__ == "__main__":
    svg() if "--svg" in sys.argv else values()
