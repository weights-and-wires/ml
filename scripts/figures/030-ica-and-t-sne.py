#!/usr/bin/env python3
"""Numbers for the ICA and t-SNE figures.

Figure 1 plots 150 whitened points made by mixing two independent uniform
sources, with the two ICA directions (the sides of the square) and a rotated
pair of axes that PCA could equally have returned.
Figure 2 plots the excess kurtosis of the whitened projection against the
angle of the projection, for the uniform pair and for a Gaussian pair.
Figure 3 plots the Gaussian and Student-t kernels against distance, and a
t-SNE map of three planted clusters.

Prints the values drawn. With --svg DIR it also writes fig1.svg, fig2.svg,
and fig3.svg into DIR. Stdlib and numpy only.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def whiten(X):
    Xc = X - X.mean(axis=0)
    lam, U = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    return Xc @ U / np.sqrt(lam)


def excess_kurtosis(u):
    u = (u - u.mean()) / u.std()
    return float(np.mean(u**4) - 3.0)


def fastica_dir(Xw, w, steps=100):
    for _ in range(steps):
        t = np.tanh(Xw @ w)
        w = Xw.T @ t / len(Xw) - np.mean(1 - t**2) * w
        w /= np.linalg.norm(w)
    return w


def sq_dists(X):
    sq = (X**2).sum(1)
    D = np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0)
    np.fill_diagonal(D, 0)
    return D


def tsne(X, perp=15.0, steps=300, lr=100.0):
    D = sq_dists(X)
    n = len(D)
    P = np.zeros((n, n))
    for i in range(n):
        row = np.delete(D[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(200):
            z = -row * beta
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
            if abs(H - np.log2(perp)) < 1e-6:
                break
            if H > np.log2(perp):
                lo = beta
                beta = beta * 2 if np.isinf(hi) else (lo + hi) / 2
            else:
                hi = beta
                beta = (lo + hi) / 2
        P[i, np.arange(n) != i] = p
    P = (P + P.T) / (2 * n)
    Y = 1e-2 * np.random.default_rng(1).normal(size=(n, 2))
    v = np.zeros_like(Y)
    for t in range(steps):
        s = 12.0 if t < 100 else 1.0
        d2 = sq_dists(Y)
        inv = 1 / (1 + d2)
        np.fill_diagonal(inv, 0)
        Q = inv / inv.sum()
        M = (s * P - Q) * inv
        g = 4 * (M.sum(1)[:, None] * Y - M @ Y)
        v = 0.8 * v - lr * g
        Y = Y + v
        Y -= Y.mean(0)
    return Y


def svg_open(vb, label):
    return f'<svg viewBox="{vb}" role="img" aria-label="{label}">'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", default=None)
    args = ap.parse_args()
    out = {}

    # Figure 1
    rng = np.random.default_rng(3)
    S = rng.uniform(-1, 1, size=(150, 2)) * np.sqrt(3)
    A = np.array([[1.0, 0.6], [0.3, 1.0]])
    Xw = whiten(S @ A.T)
    Wica = fastica_dir(Xw, np.array([1.0, 0.2]))
    ang_ica = np.degrees(np.arctan2(Wica[1], Wica[0])) % 90
    print("fig1: ICA direction angle mod 90 =", round(ang_ica, 1), "deg; cov of whitened =", np.round(np.cov(Xw.T), 3).tolist())
    c1 = abs(np.corrcoef(Xw @ Wica, S[:, 0])[0, 1]), abs(np.corrcoef(Xw @ Wica, S[:, 1])[0, 1])
    print("fig1: |corr| of the ICA output with sources =", np.round(c1, 3))
    cx, cy, sc = 200.0, 150.0, 55.0
    pts = []
    for x, y in Xw:
        pts.append(f'<circle cx="{cx + sc * x:.1f}" cy="{cy - sc * y:.1f}" r="2.6" class="fill-a"/>')
    th = np.radians(ang_ica)
    ica = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    pca = np.array([[np.cos(th + 0.6), np.sin(th + 0.6)], [-np.sin(th + 0.6), np.cos(th + 0.6)]])
    def axis(v, cls, label, dashed):
        L = 2.9 * sc
        x1, y1, x2, y2 = cx - L * v[0], cy + L * v[1], cx + L * v[0], cy - L * v[1]
        d = " dashed" if dashed else ""
        return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}{d}" stroke-width="2.5"/>'
                f'<text x="{x2 + 4:.1f}" y="{y2:.1f}" class="fig-small">{label}</text>')
    f1 = [svg_open("0 0 600 300", "Whitened points that fill a tilted square, with the two ICA axes along its sides and another pair of axes rotated away")]
    f1 += pts
    f1 += [axis(ica[0], "stroke-b", "ICA 1", False), axis(ica[1], "stroke-b", "ICA 2", False),
           axis(pca[0], "stroke-muted", "other axis", True), axis(pca[1], "stroke-muted", "", True)]
    f1.append('<text x="400" y="110" class="fig-small">covariance is the identity,</text>')
    f1.append('<text x="400" y="128" class="fig-small">so every rotation is a valid whitening.</text>')
    f1.append('<text x="400" y="160" class="fig-small">only the ICA axes run along</text>')
    f1.append('<text x="400" y="178" class="fig-small">the sides of the square.</text></svg>')
    out["fig1.svg"] = "\n".join(f1)

    # Figure 2
    rng = np.random.default_rng(0)
    n = 200000
    U = rng.uniform(-1, 1, size=(n, 2)) * np.sqrt(3)
    G = rng.normal(size=(n, 2))
    angles = np.arange(0, 91, 3)
    ku, kg = [], []
    for a in angles:
        w = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        ku.append(excess_kurtosis(U @ w))
        kg.append(excess_kurtosis(G @ w))
    print("fig2: excess kurtosis, uniform pair at 0, 45, 90 deg =", round(ku[0], 3), round(ku[15], 3), round(ku[30], 3))
    print("fig2: excess kurtosis, Gaussian pair, min and max =", round(min(kg), 3), round(max(kg), 3))
    x0, x1, y0, y1 = 60.0, 440.0, 40.0, 230.0
    def px(a): return x0 + (x1 - x0) * a / 90
    def py(k): return y0 + (y1 - y0) * (0.1 - k) / 1.4
    f2 = [svg_open("0 0 600 290", "Excess kurtosis of a whitened projection against its angle: the uniform pair swings between -1.2 and -0.6, the Gaussian pair stays at 0")]
    for k in (0.0, -0.6, -1.2):
        f2.append(f'<line x1="{x0}" y1="{py(k):.1f}" x2="{x1}" y2="{py(k):.1f}" class="stroke-grid"/><text x="{x0-8}" y="{py(k)+4:.1f}" class="fig-small" text-anchor="end">{k:g}</text>')
    for a in (0, 45, 90):
        f2.append(f'<text x="{px(a):.1f}" y="256" class="fig-small" text-anchor="middle">{a}°</text>')
    f2.append('<polyline points="' + " ".join(f"{px(a):.1f},{py(k):.1f}" for a, k in zip(angles, ku)) + '" fill="none" class="stroke-b" stroke-width="2.5"/>')
    f2.append('<polyline points="' + " ".join(f"{px(a):.1f},{py(k):.1f}" for a, k in zip(angles, kg)) + '" fill="none" class="stroke-a" stroke-width="2.5"/>')
    f2.append(f'<text x="448" y="{py(ku[15]) - 2:.1f}" class="fig-small">uniform pair</text>')
    f2.append(f'<text x="448" y="{py(0.0) + 4:.1f}" class="fig-small">Gaussian pair</text>')
    f2.append('<text x="250" y="278" text-anchor="middle">angle of the projection direction</text>')
    f2.append('<text transform="rotate(-90 14 135)" x="14" y="135" text-anchor="middle">excess kurtosis</text></svg>')
    out["fig2.svg"] = "\n".join(f2)

    # Figure 3
    d = np.linspace(0, 4, 41)
    gk, tk = np.exp(-d**2 / 2), 1 / (1 + d**2)
    print("fig3: Gaussian vs Student-t kernel at d = 1, 2, 3, 4:", [(round(float(np.exp(-x**2 / 2)), 4), round(1 / (1 + x**2), 4)) for x in (1, 2, 3, 4)])
    rng = np.random.default_rng(0)
    cent = rng.normal(size=(3, 10)) * 4.0
    X = np.vstack([c + rng.normal(size=(30, 10)) for c in cent])
    lab = np.repeat(range(3), 30)
    Y = tsne(X)
    d2 = sq_dists(Y)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, 1)[:, :5]
    print("fig3: 5-nn label agreement in the map =", float((lab[nn] == lab[:, None]).mean()))
    print("fig3: map spread (std of x, y) =", np.round(Y.std(0), 2).tolist())
    lo, hi = Y.min(0), Y.max(0)
    f3 = [svg_open("0 0 600 280", "Left: the Gaussian kernel falls faster than the Student-t kernel with one degree of freedom. Right: a t-SNE map of three planted clusters")]
    gx0, gx1, gy0, gy1 = 50.0, 270.0, 30.0, 220.0
    f3.append(f'<line x1="{gx0}" y1="{gy1}" x2="{gx1}" y2="{gy1}" class="stroke-grid"/><line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy1}" class="stroke-grid"/>')
    f3.append('<polyline points="' + " ".join(f"{gx0 + (gx1-gx0)*x/4:.1f},{gy1 - (gy1-gy0)*y:.1f}" for x, y in zip(d, gk)) + '" fill="none" class="stroke-a" stroke-width="2.5"/>')
    f3.append('<polyline points="' + " ".join(f"{gx0 + (gx1-gx0)*x/4:.1f},{gy1 - (gy1-gy0)*y:.1f}" for x, y in zip(d, tk)) + '" fill="none" class="stroke-b" stroke-width="2.5"/>')
    f3.append('<line x1="130" y1="50" x2="160" y2="50" class="stroke-a" stroke-width="2.5"/><text x="166" y="54" class="fig-small">Gaussian: exp(-d²/2)</text>')
    f3.append('<line x1="130" y1="72" x2="160" y2="72" class="stroke-b" stroke-width="2.5"/><text x="166" y="76" class="fig-small">Student-t: 1/(1+d²)</text>')
    f3.append(f'<text x="{(gx0+gx1)/2:.1f}" y="256" text-anchor="middle">distance d</text>')
    mx0, mx1, my0, my1 = 330.0, 580.0, 30.0, 230.0
    cls = ["fill-a", "fill-b", "fill-c"]
    for (x, y), l in zip(Y, lab):
        f3.append(f'<circle cx="{mx0 + (mx1-mx0)*(x-lo[0])/(hi[0]-lo[0]):.1f}" cy="{my1 - (my1-my0)*(y-lo[1])/(hi[1]-lo[1]):.1f}" r="3.4" class="{cls[l]}"/>')
    f3.append('<text x="455" y="256" text-anchor="middle">t-SNE map, three planted clusters</text></svg>')
    out["fig3.svg"] = "\n".join(f3)

    if args.svg:
        os.makedirs(args.svg, exist_ok=True)
        for name, text in out.items():
            with open(os.path.join(args.svg, name), "w") as f:
                f.write(text)


if __name__ == "__main__":
    main()
