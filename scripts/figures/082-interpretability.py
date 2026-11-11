#!/usr/bin/env python3
"""Numbers and SVG for the interpretability lesson.

Figure 1 shows probe accuracy on planted activations, a random network, and
shuffled labels (the data of the exercise's probe check).
Figure 2 compares three attributions of the worked model f(x) = x0*x1 + 2*x2
at x = [2, 3, 1] with a zero baseline.
Figure 3 shows three planted directions in two dimensions, the sparse
activations they make, and the directions a sparse autoencoder learns.

Run it with `python scripts/figures/082-interpretability.py`. It prints every
value it draws. Add `--svg` to print the SVG blocks too.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# ------------------------------------------------------------- attribution
F = lambda v: v[0] * v[1] + 2.0 * v[2]
GRAD = lambda v: np.array([v[1], v[0], 2.0])
X = np.array([2.0, 3.0, 1.0])
B = np.zeros(3)


def ig(grad, x, b, steps):
    a = (np.arange(steps) + 0.5) / steps
    return (x - b) * np.mean([grad(b + t * (x - b)) for t in a], axis=0)


def shapley(f, x, b):
    n = len(x)
    val = {m: f(np.where([(m >> j) & 1 for j in range(n)], x, b)) for m in range(2**n)}
    phi = np.zeros(n)
    for i in range(n):
        for m in range(2**n):
            if (m >> i) & 1:
                continue
            s = bin(m).count("1")
            phi[i] += math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) * (val[m | 1 << i] - val[m])
    return phi


# ------------------------------------------------------------------ probes
def sig(z):
    return 1 / (1 + np.exp(-z))


def train_probe(H, y, steps=500, lr=0.5, l2=1e-3):
    w, b = np.zeros(H.shape[1]), 0.0
    for _ in range(steps):
        r = sig(H @ w + b) - y
        w = w - lr * (H.T @ r / len(y) + l2 * w)
        b = b - lr * r.mean()
    return w, b


def probe_numbers():
    rng = np.random.default_rng(0)
    n = 800
    Xp = rng.normal(size=(n, 6))
    y = (Xp[:, 0] * Xp[:, 1] > 0).astype(float)
    Wp, Wr = rng.normal(size=(6, 7)), rng.normal(size=(6, 8))
    Hr = np.maximum(Xp @ Wr, 0.0)
    unit = np.tanh(2 * Xp[:, 0]) * np.tanh(2 * Xp[:, 1])
    Hp = np.concatenate([np.maximum(Xp @ Wp, 0.0), unit[:, None]], axis=1)
    tr, te = slice(0, 500), slice(500, n)

    def held(H, lab):
        w, b = train_probe(H[tr], lab[tr])
        return float(np.mean((H[te] @ w + b > 0) == (lab[te] > 0.5)))

    return {"planted": held(Hp, y), "random network": held(Hr, y), "shuffled labels": held(Hp, rng.permutation(y))}


# --------------------------------------------------------------------- SAE
def sae_train(Xd, k, l1, lr, steps, seed):
    rng = np.random.default_rng(seed)
    Wd = rng.normal(size=(k, Xd.shape[1]))
    Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    We, be = Wd.T.copy(), np.zeros(k)
    n = len(Xd)
    for _ in range(steps):
        pre = Xd @ We + be
        f = np.maximum(pre, 0)
        r = f @ Wd - Xd
        dxh = 2 * r / n
        dpre = (dxh @ Wd.T + l1 / n * (pre > 0)) * (pre > 0)
        gWd = f.T @ dxh
        We, be, Wd = We - lr * (Xd.T @ dpre), be - lr * dpre.sum(0), Wd - lr * gWd
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    return Wd


def sae_numbers():
    rng = np.random.default_rng(0)
    ang = np.deg2rad([90.0, 210.0, 330.0])
    D = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # three true directions in 2 dims
    n = 400
    mask = rng.uniform(size=(n, 3)) < 0.12
    Xd = (mask * 1.0) @ D
    Wd = sae_train(Xd, 3, 0.1, 0.2, 3000, 1)
    cos = np.abs(D @ Wd.T)
    return D, Xd, mask, Wd, cos.max(axis=1)


def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def fig1(phi_gi, phi_ig, phi_sh):
    rows = [("gradient × input", phi_gi, "fill-b"), ("integrated gradients", phi_ig, "fill-a"), ("exact Shapley", phi_sh, "fill-c")]
    out = ['<svg viewBox="0 0 600 260" role="img" aria-label="Bars for three attribution methods on the worked model; gradient times input sums to 14, the others to 8">']
    unit = 26.0
    x0 = 170.0
    out.append(f'<line x1="{x0}" y1="14" x2="{x0}" y2="206" class="stroke-muted"/>')
    for r, (name, phi, cls) in enumerate(rows):
        y0 = 22 + r * 62
        out.append(f'<text x="{x0 - 10}" y="{y0 + 30}" text-anchor="end" class="fig-bold">{name}</text>')
        for i, v in enumerate(phi):
            y = y0 + i * 16
            w = v * unit
            out.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="13" class="{cls}"/>')
            out.append(f'<text x="{x0 + w + 6:.1f}" y="{y + 11}" class="fig-small">x{i}: {v:g}</text>')
        out.append(f'<text x="{x0 + 8 * unit + 40:.1f}" y="{y0 + 30}" class="fig-small">sum {phi.sum():g}</text>')
    xt = x0 + 8 * unit
    out.append(f'<line x1="{xt}" y1="14" x2="{xt}" y2="206" class="stroke-b dashed"/>')
    out.append(f'<text x="{xt}" y="226" text-anchor="middle" class="fig-small">f(x) - f(baseline) = 8</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig2(acc):
    items = list(acc.items())
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Held-out probe accuracy: planted activations, random network, shuffled labels">']
    out.append('<line x1="60" y1="200" x2="580" y2="200" class="stroke-muted"/>')
    out.append('<line x1="60" y1="30" x2="60" y2="200" class="stroke-muted"/>')
    for t in (0.0, 0.5, 1.0):
        y = 200 - t * 170
        out.append(f'<line x1="56" y1="{y:.0f}" x2="580" y2="{y:.0f}" class="stroke-grid"/>')
        out.append(f'<text x="50" y="{y + 4:.0f}" text-anchor="end" class="fig-small">{t:g}</text>')
    for i, (name, v) in enumerate(items):
        x = 110 + i * 160
        h = v * 170
        cls = "fill-a" if i == 0 else "fill-muted"
        out.append(f'<rect x="{x}" y="{200 - h:.1f}" width="90" height="{h:.1f}" class="{cls}"/>')
        out.append(f'<text x="{x + 45}" y="{194 - h:.1f}" text-anchor="middle" class="fig-bold">{v:.2f}</text>')
        out.append(f'<text x="{x + 45}" y="222" text-anchor="middle" class="fig-small">{name}</text>')
    out.append('<text x="20" y="22" class="fig-small">held-out accuracy of the same logistic probe</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig3(D, Xd, Wd):
    cx, cy, s = 300.0, 150.0, 105.0
    out = ['<svg viewBox="0 0 600 310" role="img" aria-label="Three feature directions in a plane, the points they make, and the directions a sparse autoencoder learns">']
    out.append('<defs><marker id="c066-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-muted"/></marker></defs>')
    out.append(f'<line x1="{cx - 150}" y1="{cy}" x2="{cx + 150}" y2="{cy}" class="stroke-grid"/>')
    out.append(f'<line x1="{cx}" y1="{cy - 140}" x2="{cx}" y2="{cy + 140}" class="stroke-grid"/>')
    seen = set()
    for p in Xd:
        key = (round(p[0], 3), round(p[1], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(f'<circle cx="{cx + p[0] * s:.1f}" cy="{cy - p[1] * s:.1f}" r="3.5" class="fill-soft-a stroke-a"/>')
    for d in D:
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + d[0] * s * 1.12:.1f}" y2="{cy - d[1] * s * 1.12:.1f}" class="stroke-ink" marker-end="url(#c066-arrow)"/>')
    for w in Wd:
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + w[0] * s * 0.9:.1f}" y2="{cy - w[1] * s * 0.9:.1f}" class="stroke-b dashed" stroke-width="2.5"/>')
    out.append('<text x="14" y="26" class="fig-small">black arrows: planted directions</text>')
    out.append('<text x="14" y="44" class="fig-small">orange dashes: learned directions</text>')
    out.append('<text x="14" y="62" class="fig-small">blue dots: activations</text>')
    out.append(f'<text x="{cx}" y="{cy + 140}" text-anchor="middle" class="fig-small">three features in two dimensions</text>')
    out.append("</svg>")
    return "\n".join(out)


def ig_errors():
    a = np.random.default_rng(0).normal(size=5)
    f = lambda v: np.tanh(a @ v) + 0.5 * np.sum(v * v)
    g = lambda v: (1 - np.tanh(a @ v) ** 2) * a + v
    rng = np.random.default_rng(1)
    x, b = rng.normal(size=5), rng.normal(size=5)
    return {s: abs(float(ig(g, x, b, s).sum()) - (f(x) - f(b))) for s in (2, 8, 32, 128)}


def sae_seeds():
    out = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        D = rng.normal(size=(16, 8))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        Xd = ((rng.uniform(size=(3000, 16)) < 0.04) * 1.0) @ D
        Wd = sae_train(Xd, 16, 0.3, 0.2, 4000, 100 + seed)
        R = rng.normal(size=(16, 8))
        score = lambda W: float(np.abs(D @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T).max(axis=1).mean())
        out.append((round(score(Wd), 3), round(score(R), 3)))
    return out


def main():
    phi_gi = GRAD(X) * X
    phi_ig = ig(GRAD, X, B, 4)
    phi_sh = shapley(F, X, B)
    print("f(x) - f(0) =", F(X) - F(B))
    print("grad x input", phi_gi, "sum", phi_gi.sum())
    print("integrated gradients", phi_ig, "sum", phi_ig.sum())
    print("exact shapley", phi_sh, "sum", phi_sh.sum())
    acc = probe_numbers()
    print("probe accuracy", {k: round(v, 3) for k, v in acc.items()})
    D, Xd, mask, Wd, best = sae_numbers()
    print("SAE: true directions", np.round(D, 3).tolist())
    print("SAE: learned directions", np.round(Wd, 3).tolist())
    print("SAE: best cosine per true direction", np.round(best, 3), "mean", round(float(best.mean()), 3))
    print("SAE: share of rows with 0, 1, 2+ active features", [round(float(np.mean(mask.sum(1) == k)), 3) for k in (0, 1)], round(float(np.mean(mask.sum(1) >= 2)), 3))
    print("IG completeness error by steps (tanh model):", {k: f"{v:.2e}" for k, v in ig_errors().items()})
    print("SAE 16 features in 8 dims, (trained, random dictionary) over 5 seeds:", sae_seeds())
    if "--svg" in sys.argv:
        print(fig1(phi_gi, phi_ig, phi_sh)); print(fig2(acc)); print(fig3(D, Xd, Wd))


if __name__ == "__main__":
    main()
