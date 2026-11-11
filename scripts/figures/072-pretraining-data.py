#!/usr/bin/env python3
"""Numbers and SVG for the pretraining-data figures.

Figure 1 is the hand MinHash example: three hashes over six shingles.
Figure 3 is the S-curve 1 - (1 - s^r)^b for two ways to spend 128 hashes
(r = 4, b = 32 and r = 8, b = 16), with Monte Carlo dots on the first curve.
Figure 2 is the root-mean-square error of the MinHash estimate at a true
Jaccard similarity of 0.5, measured over 400 seeded pairs for each signature
length k, against the theory sqrt(s (1 - s) / k).
Figure 4 is the temperature-sampling share of three domains at T = 1, 2, 4,
and a very large T.

Run with no argument to print the values. Run with `--svg 1` to `--svg 4` to print the SVG that the lesson shows. Re-run this file before
changing a figure.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

PRIME = 2_147_483_647


def stable_id(s: str) -> int:
    d = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") % PRIME


def signature(items, k, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=k, dtype=np.int64)
    b = rng.integers(0, PRIME, size=k, dtype=np.int64)
    ids = np.array([stable_id(x) for x in items], dtype=np.int64)
    return ((a[:, None] * ids[None, :] + b[:, None]) % PRIME).min(axis=1)


def lsh_prob(s, r, b):
    return 1 - (1 - s**r) ** b


def scurve_values():
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rng = np.random.default_rng(0)
    rows = []
    for s in grid:
        mc = float(np.mean((rng.random((20000, 32, 4)) < s).all(axis=2).any(axis=1)))
        rows.append((s, lsh_prob(s, 4, 32), lsh_prob(s, 8, 16), mc))
    return rows


def error_values():
    out = []
    for k in [16, 64, 256, 1024]:
        errs = []
        for t in range(400):
            a = [f"p{t}-{i}" for i in range(60)]
            b = [f"p{t}-{i}" for i in range(20, 80)]  # 40 shared, 80 in the union: J = 0.5
            errs.append(float(np.mean(signature(a, k, t) == signature(b, k, t))) - 0.5)
        out.append((k, float(np.sqrt(np.mean(np.square(errs)))), float(np.sqrt(0.25 / k))))
    return out


def mix_values():
    sizes = np.array([900.0, 90.0, 10.0])
    rows = []
    for T in [1, 2, 4, 1000]:
        w = sizes ** (1 / T)
        rows.append((T, w / w.sum()))
    return rows


HAND_A, HAND_B = {0, 1, 2, 3, 4}, {0, 1, 2, 3, 5}
HAND_HASHES = [lambda x: (x + 1) % 7, lambda x: (3 * x + 1) % 7, lambda x: (x + 3) % 7]
HAND_NAMES = ["(x+1) mod 7", "(3x+1) mod 7", "(x+3) mod 7"]
SHINGLE_NAMES = ["the cat", "cat sat", "sat on", "on the", "the mat", "the rug"]


def svg0():
    out = ['<svg viewBox="0 0 640 270" role="img" aria-label="Three hash functions over six shingles, with the smallest value in each document outlined and matches in green">']
    x0, cw = 130, 76
    for j, name in enumerate(SHINGLE_NAMES):
        out.append(f'<text x="{x0 + j * cw + cw / 2 - 4:.0f}" y="22" text-anchor="middle" class="fig-small">{name}</text>')
        member = ("A" if j in HAND_A else "") + ("B" if j in HAND_B else "")
        out.append(f'<text x="{x0 + j * cw + cw / 2 - 4:.0f}" y="40" text-anchor="middle" class="fig-small fig-bold">in {" and ".join(member)}</text>')
    for i, (h, name) in enumerate(zip(HAND_HASHES, HAND_NAMES)):
        y = 56 + i * 56
        out.append(f'<text x="{x0 - 10}" y="{y + 26}" text-anchor="end" class="fig-small fig-mono">{name}</text>')
        vals = [h(x) for x in range(6)]
        ma = min(vals[x] for x in HAND_A)
        mb = min(vals[x] for x in HAND_B)
        ia = [x for x in HAND_A if vals[x] == ma][0]
        ib = [x for x in HAND_B if vals[x] == mb][0]
        for j in range(6):
            cls = "fill-sunk"
            stroke = "stroke-grid"
            if j == ia and j == ib:
                cls, stroke = "fill-soft-c", "stroke-c"
            elif j == ia:
                cls, stroke = "fill-soft-a", "stroke-a"
            elif j == ib:
                cls, stroke = "fill-soft-b", "stroke-b"
            out.append(f'<rect x="{x0 + j * cw}" y="{y}" width="{cw - 8}" height="40" rx="4" class="{cls} {stroke}" stroke-width="2"/>')
            out.append(f'<text x="{x0 + j * cw + (cw - 8) / 2:.0f}" y="{y + 26}" text-anchor="middle" class="fig-mono">{vals[j]}</text>')
        out.append(f'<text x="{x0 + 6 * cw + 4}" y="{y + 26}" class="fig-small fig-bold">{"match" if ma == mb else "miss"}</text>')
    out.append('<text x="20" y="236" class="fig-small">Blue: smallest over A. Orange: smallest over B. Green: the same shingle for both.</text>')
    out.append('<text x="20" y="256" class="fig-small">Matches: 2 of 3, so the estimate is 0.667; the true J is 4/6 = 0.667.</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg1():
    x0, y0, w, h = 60, 20, 440, 230  # plot box
    px = lambda s: x0 + s * w
    py = lambda p: y0 + h - p * h
    out = ['<svg viewBox="0 0 600 300" role="img" aria-label="S-curves of the chance that two documents share a bucket, for 4 rows by 32 bands and 8 rows by 16 bands">']
    for p in [0, 0.5, 1]:
        out.append(f'<line x1="{x0}" y1="{py(p):.1f}" x2="{x0 + w}" y2="{py(p):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{py(p) + 4:.1f}" text-anchor="end" class="fig-small">{p:g}</text>')
    for s in [0, 0.25, 0.5, 0.75, 1]:
        out.append(f'<text x="{px(s):.1f}" y="{y0 + h + 18}" text-anchor="middle" class="fig-small">{s:g}</text>')
    out.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 40}" text-anchor="middle">true Jaccard similarity s</text>')
    out.append(f'<text x="14" y="{y0 + h / 2}" text-anchor="middle" transform="rotate(-90 14 {y0 + h / 2})">P(candidate pair)</text>')
    for (r, b, cls) in [(4, 32, "stroke-a"), (8, 16, "stroke-b")]:
        pts = " ".join(f"{px(s):.1f},{py(lsh_prob(s, r, b)):.1f}" for s in np.linspace(0, 1, 101))
        out.append(f'<polyline points="{pts}" fill="none" class="{cls}" stroke-width="2.5"/>')
    for s, _, _, mc in scurve_values():
        out.append(f'<circle cx="{px(s):.1f}" cy="{py(mc):.1f}" r="4" class="fill-ink"/>')
    out.append(f'<text x="{px(0.02):.1f}" y="{py(0.62):.1f}" class="fig-small fig-bold fill-a">r = 4, b = 32</text>')
    out.append(f'<text x="{px(0.74):.1f}" y="{py(0.4):.1f}" class="fig-small fig-bold fill-b">r = 8, b = 16</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg2():
    rows = error_values()
    vmax = rows[0][1]
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Bars of the measured and theoretical error of the MinHash estimate for 16, 64, 256, and 1024 hashes">']
    for i, (k, meas, theory) in enumerate(rows):
        y = 20 + i * 56
        out.append(f'<text x="70" y="{y + 20}" text-anchor="end">k = {k}</text>')
        out.append(f'<rect x="80" y="{y}" width="{meas / vmax * 380:.1f}" height="20" class="fill-a"/>')
        out.append(f'<text x="{80 + meas / vmax * 380 + 8:.1f}" y="{y + 15}" class="fig-small fig-mono">{meas:.3f}</text>')
        out.append(f'<rect x="80" y="{y + 24}" width="{theory / vmax * 380:.1f}" height="12" class="fill-muted"/>')
        out.append(f'<text x="{80 + theory / vmax * 380 + 8:.1f}" y="{y + 35}" class="fig-small fig-mono">{theory:.3f}</text>')
    out.append('<rect x="440" y="196" width="14" height="14" class="fill-a"/><text x="460" y="208" class="fig-small">measured</text>')
    out.append('<rect x="440" y="218" width="14" height="10" class="fill-muted"/><text x="460" y="228" class="fig-small">sqrt(0.25 / k)</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg3():
    rows = mix_values()
    names = ["web 900", "code 90", "books 10"]
    cls = ["fill-a", "fill-b", "fill-c"]
    out = ['<svg viewBox="0 0 600 240" role="img" aria-label="Stacked bars of the sampling share of web, code, and books at T = 1, 2, 4, and a very large T">']
    for i, (T, w) in enumerate(rows):
        y = 20 + i * 46
        label = "T = 1" if T == 1 else f"T = {T}" if T < 100 else "T large"
        out.append(f'<text x="70" y="{y + 20}" text-anchor="end">{label}</text>')
        x = 80.0
        for j in range(3):
            wd = w[j] * 440
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{wd:.1f}" height="28" class="{cls[j]}"/>')
            if wd > 34:
                out.append(f'<text x="{x + wd / 2:.1f}" y="{y + 19}" text-anchor="middle" class="fig-small fill-paper">{w[j]:.2f}</text>')
            x += wd
    for j in range(3):
        out.append(f'<rect x="{90 + j * 150}" y="208" width="14" height="14" class="{cls[j]}"/><text x="{110 + j * 150}" y="220" class="fig-small">{names[j]}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--svg":
        print({"1": svg0, "2": svg2, "3": svg1, "4": svg3}[sys.argv[2]]())
        sys.exit(0)
    print("figure 3: s, 1-(1-s^4)^32, 1-(1-s^8)^16, Monte Carlo r=4 b=32")
    for s, a, b, mc in scurve_values():
        print(f"  {s:.1f}  {a:.4f}  {b:.4f}  {mc:.4f}")
    print("knee (1/b)^(1/r):", round((1 / 32) ** 0.25, 3), round((1 / 16) ** 0.125, 3))
    print("figure 2: k, measured rms error, theory sqrt(0.25 / k)")
    for k, m, t in error_values():
        print(f"  {k:5d}  {m:.4f}  {t:.4f}")
    print("figure 4: T, shares of web, code, books for sizes 900, 90, 10")
    for T, w in mix_values():
        print(f"  {T:5d}  {np.round(w, 4)}")
    sizes = np.array([900.0, 90.0, 10.0])
    w2 = sizes**0.5 / (sizes**0.5).sum()
    print("epochs at T = 2, budget 300:", np.round(300 * w2 / sizes, 4))
    print("epochs at T = 1, budget 300:", np.round(300 * sizes / sizes.sum() / sizes, 4))
    # hand example: universe ids 0..5, hashes (x+1)%7, (3x+1)%7, (x+3)%7
    A, B = {0, 1, 2, 3, 4}, {0, 1, 2, 3, 5}
    for name, f in [("(x+1)%7", lambda x: (x + 1) % 7), ("(3x+1)%7", lambda x: (3 * x + 1) % 7), ("(x+3)%7", lambda x: (x + 3) % 7)]:
        print("hand minhash", name, min(map(f, A)), min(map(f, B)))
    print("J =", len(A & B) / len(A | B))
