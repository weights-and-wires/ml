#!/usr/bin/env python3
"""Numbers and SVG for the diffusion language model lesson.

Figure 1 masks one 8-word sentence at t = 0, 0.25, 0.5, 0.75, 1 with one
uniform draw per position, so each row's masks sit inside the next row's.
Figure 2 runs the low-confidence sampler and a random sampler on the 8-token
table of logits that the exercise tests use.

The script prints the worked loss, the schedule, the confidences, both
histories, and the two SVG blocks that the lesson embeds.
"""

from __future__ import annotations

import numpy as np

MASK = 4
WORDS = ["the", "cat", "sat", "on", "the", "mat", "all", "day"]
TABLE = np.array(
    [[3, 0, 0, 0], [0, 0, 5, 0], [0, 2, 0, 0], [0, 0, 0, 4],
     [1, 0, 0, 0], [0, 0, 2.5, 0], [0, 1.5, 0, 0], [0, 0, 0, 0.5]],
    dtype=float,
)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def step(x, logits, n_unmasked, rng=None):
    probs = softmax(logits)
    pred, conf = probs.argmax(-1), probs.max(-1)
    masked = np.flatnonzero(x == MASK)
    n_commit = max(0, n_unmasked - (len(x) - len(masked)))
    if rng is None:
        order = masked[np.argsort(-conf[masked], kind="stable")]
    else:
        order = rng.permutation(masked)
    out = x.copy()
    out[order[:n_commit]] = pred[order[:n_commit]]
    return out


def run(rng=None, n_steps=4):
    x = np.full(8, MASK)
    hist = [x.copy()]
    for k in range(1, n_steps + 1):
        x = step(x, TABLE, (8 * k) // n_steps, rng)
        hist.append(x.copy())
    return hist


def cell(x, y, w, h, label, masked):
    fill = "fill-sunk" if masked else "fill-soft-a"
    cls = "fill-muted fig-small" if masked else "fig-small"
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" class="{fill} stroke-grid"/>\n'
        f'  <text x="{x + w / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" class="{cls}">{label}</text>'
    )


def fig1():
    u = np.random.default_rng(0).uniform(size=8)
    print("uniform draws:", np.round(u, 3).tolist())
    rows = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="One sentence of eight words masked at five levels of t; masked positions stay masked as t grows">']
    for r, t in enumerate(rows):
        m = u < t
        print(f"t={t}: masked={int(m.sum())} positions {np.flatnonzero(m).tolist()}")
        y = 20 + r * 44
        out.append(f'  <text x="10" y="{y + 22}" class="fig-small">t = {t:g}</text>')
        for i, w in enumerate(WORDS):
            out.append(cell(90 + i * 62, y, 58, 34, "M" if m[i] else w, bool(m[i])))
    out.append("</svg>")
    return "\n".join(out)


def fig2():
    probs = softmax(TABLE)
    conf = probs.max(-1)
    lc, rd = run(), run(np.random.default_rng(0))
    print("confidence:", np.round(conf, 3).tolist())
    print("low-confidence history:")
    for h in lc:
        print(" ", h.tolist())
    print("random history (seed 0):")
    for h in rd:
        print(" ", h.tolist())
    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="Five rows of eight tokens for each sampler: low-confidence fills the high-confidence slots first, random fills slots in arbitrary order">']
    for p, (name, hist, x0) in enumerate([("low-confidence", lc, 60), ("random", rd, 340)]):
        out.append(f'  <text x="{x0 + 100}" y="18" text-anchor="middle" class="fig-bold">{name}</text>')
        out.append(f'  <text x="{x0 - 4}" y="46" text-anchor="end" class="fig-small">conf</text>')
        for i in range(8):
            out.append(f'  <text x="{x0 + i * 25 + 12}" y="46" text-anchor="middle" class="fig-small">{conf[i]:.2f}</text>')
        for r, h in enumerate(hist):
            y = 56 + r * 40
            out.append(f'  <text x="{x0 - 4}" y="{y + 24}" text-anchor="end" class="fig-small">step {r}</text>')
            for i in range(8):
                out.append(cell(x0 + i * 25, y, 24, 34, "M" if h[i] == MASK else int(h[i]), bool(h[i] == MASK)))
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    # Worked loss: L = 4, t = 0.5, masked positions 1 and 3.
    p = np.array([[.2, .3, .5], [.5, .25, .25], [.1, .8, .1], [.25, .5, .25]])
    tg = np.array([2, 0, 1, 0])
    mask = np.array([0, 1, 0, 1], bool)
    loss = -np.sum(mask * np.log(p[np.arange(4), tg])) / (0.5 * 4)
    allpos = -np.mean(np.log(p[np.arange(4), tg]))
    print(f"hand loss = {loss:.6f}; mean over all positions = {allpos:.6f}")
    print("unmasked after k=1..4 (L=8):", [(8 * k) // 4 for k in range(1, 5)])
    print("unmasked after k=1..4 (L=10):", [(10 * k) // 4 for k in range(1, 5)])
    svg1 = fig1()
    svg2 = fig2()
    print("\nFIGURE 1\n" + svg1)
    print("\nFIGURE 2\n" + svg2)
