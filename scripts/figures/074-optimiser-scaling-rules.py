#!/usr/bin/env python3
"""Numbers and SVG coordinates for the optimiser-scaling-rules lesson.

Figure 1 draws how far the last hidden layer moves after one Adam step, against
width, under standard parameterisation (SP) and muP.
Figure 2 draws the final loss against the learning rate at widths 32, 128, and
512, under SP and under muP, and marks the best rate in each curve.
Figure 3 draws the singular value that one, and five, Newton-Schulz steps
return, for a singular value that goes in between 0 and 1.

The script also prints the worked arithmetic quoted in the lesson. Run it with
`--svg DIR` to write the three SVG bodies into DIR as fig1.svg to fig3.svg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

D_IN = 16
BASE_WIDTH = 32
A, B, C = 3.4445, -4.7750, 2.0315  # Jordan (2024), the tuned quintic


# ---------------------------------------------------------------- the network
def init(widths, rng, mup):
    layers = len(widths) - 1
    out = []
    for i in range(layers):
        fan_in, fan_out = widths[i], widths[i + 1]
        std = 1.0 / np.sqrt(fan_in)
        if mup and i == layers - 1:
            std = 1.0 / fan_in
        out.append(rng.normal(0.0, std, (fan_out, fan_in)))
    return out


def forward_backward(Ws, x, y):
    acts, pre = [x.T], []
    a = x.T
    for i, W in enumerate(Ws):
        z = W @ a
        pre.append(z)
        a = z if i == len(Ws) - 1 else np.maximum(z, 0.0)
        acts.append(a)
    err = acts[-1] - y.T
    loss = 0.5 * np.mean(err**2)
    delta = err / x.shape[0]
    grads = [None] * len(Ws)
    for i in reversed(range(len(Ws))):
        grads[i] = delta @ acts[i].T
        if i > 0:
            delta = (Ws[i].T @ delta) * (pre[i - 1] > 0)
    return loss, grads, pre


def layer_lrs(n, lr, mup):
    return [lr if (not mup or i == 0) else lr * BASE_WIDTH / n for i in range(3)]


def one_step_size(n, mup, seed, lr=1e-3):
    rng = np.random.default_rng(seed)
    widths = [D_IN, n, n, 1]
    Ws = init(widths, rng, mup)
    x, y = rng.normal(size=(8, D_IN)), rng.normal(size=(8, 1))
    _, g, pre = forward_backward(Ws, x, y)
    W2 = [W - l * gi / (np.abs(gi) + 1e-8) for W, l, gi in zip(Ws, layer_lrs(n, lr, mup), g)]
    _, _, pre2 = forward_backward(W2, x, y)
    return float(np.sqrt(np.mean((pre2[1] - pre[1]) ** 2)))


def train_loss(n, mup, lr, seed, steps=200):
    rng = np.random.default_rng(seed)
    widths = [D_IN, n, n, 1]
    Ws = init(widths, rng, mup)
    t_rng = np.random.default_rng(123)
    T1 = t_rng.normal(size=(8, D_IN)) / np.sqrt(D_IN)
    T2 = t_rng.normal(size=(1, 8))
    lrs = layer_lrs(n, lr, mup)
    m = [np.zeros_like(W) for W in Ws]
    v = [np.zeros_like(W) for W in Ws]
    d_rng = np.random.default_rng(7)
    for t in range(1, steps + 1):
        x = d_rng.normal(size=(64, D_IN))
        _, g, _ = forward_backward(Ws, x, np.tanh(x @ T1.T) @ T2.T)
        for i in range(3):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            Ws[i] = Ws[i] - lrs[i] * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    xe = d_rng.normal(size=(512, D_IN))
    return float(forward_backward(Ws, xe, np.tanh(xe @ T1.T) @ T2.T)[0])


# ---------------------------------------------------------- Newton-Schulz
def ns_scalar(s, steps, coeffs):
    a, b, c = coeffs
    for _ in range(steps):
        s = a * s + b * s**3 + c * s**5
    return s


CUBIC = (1.5, -0.5, 0.0)
QUINTIC = (A, B, C)


# ------------------------------------------------------------------- svg
class Axes:
    def __init__(self, x0, x1, y0, y1, xr, yr, xlog=False, ylog=False):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.xr, self.yr, self.xlog, self.ylog = xr, yr, xlog, ylog

    def px(self, x):
        lo, hi = self.xr
        f = (np.log2(x) - np.log2(lo)) / (np.log2(hi) - np.log2(lo)) if self.xlog else (x - lo) / (hi - lo)
        return self.x0 + f * (self.x1 - self.x0)

    def py(self, y):
        lo, hi = self.yr
        f = (np.log2(y) - np.log2(lo)) / (np.log2(hi) - np.log2(lo)) if self.ylog else (y - lo) / (hi - lo)
        return self.y1 - f * (self.y1 - self.y0)


def pts(ax, xs, ys):
    return " ".join(f"{ax.px(x):.1f},{ax.py(y):.1f}" for x, y in zip(xs, ys))


def frame(ax):
    return (f'<line x1="{ax.x0}" y1="{ax.y1}" x2="{ax.x1}" y2="{ax.y1}" class="stroke-muted"/>'
            f'<line x1="{ax.x0}" y1="{ax.y0}" x2="{ax.x0}" y2="{ax.y1}" class="stroke-muted"/>')


def fig1(sp, mup, widths):
    ax = Axes(70, 540, 50, 200, (32, 512), (0.004, 0.2), xlog=True, ylog=True)
    s = ['<svg viewBox="0 0 600 270" role="img" aria-label="Size of the change in the last hidden layer after one Adam step against width: it grows about tenfold under standard parameterisation and stays flat under muP">']
    s.append(frame(ax))
    for w in widths:
        s.append(f'<line x1="{ax.px(w):.1f}" y1="{ax.y0}" x2="{ax.px(w):.1f}" y2="{ax.y1}" class="stroke-grid"/>'
                 f'<text x="{ax.px(w):.1f}" y="{ax.y1 + 16}" text-anchor="middle" class="fig-small">{w}</text>')
    for yv, lab in [(0.01, "0.01"), (0.03, "0.03"), (0.1, "0.1")]:
        s.append(f'<line x1="{ax.x0}" y1="{ax.py(yv):.1f}" x2="{ax.x1}" y2="{ax.py(yv):.1f}" class="stroke-grid"/>'
                 f'<text x="{ax.x0 - 6}" y="{ax.py(yv) + 4:.1f}" text-anchor="end" class="fig-small">{lab}</text>')
    s.append(f'<polyline points="{pts(ax, widths, sp)}" fill="none" class="stroke-a" stroke-width="2"/>')
    s.append(f'<polyline points="{pts(ax, widths, mup)}" fill="none" class="stroke-b" stroke-width="2"/>')
    for w, a_, b_ in zip(widths, sp, mup):
        s.append(f'<circle cx="{ax.px(w):.1f}" cy="{ax.py(a_):.1f}" r="5" class="fill-a"/>'
                 f'<circle cx="{ax.px(w):.1f}" cy="{ax.py(b_):.1f}" r="5" class="fill-b"/>')
    s.append(f'<text x="{ax.px(300):.1f}" y="{ax.py(sp[3]) - 18:.1f}" class="fig-small">standard: grows with width</text>')
    s.append(f'<text x="{ax.px(300):.1f}" y="{ax.py(mup[3]) + 26:.1f}" class="fig-small">muP: flat</text>')
    s.append(f'<text x="305" y="252" text-anchor="middle" class="fig-small">width of the hidden layers</text>')
    s.append('<text x="12" y="34" class="fig-small">RMS change in a hidden activation</text>')
    s.append("</svg>")
    return "".join(s)


def fig2(curves, grid, widths):
    s = ['<svg viewBox="0 0 600 290" role="img" aria-label="Final loss against learning rate at widths 32, 128, and 512. Under standard parameterisation the best rate moves left as the width grows. Under muP the best rate stays in one place.">']
    titles = ["Standard parameterisation", "muP"]
    classes = ["stroke-a", "stroke-b", "stroke-c"]
    fills = ["fill-a", "fill-b", "fill-c"]
    for p, x0 in enumerate([55, 340]):
        ax = Axes(x0, x0 + 230, 50, 220, (float(grid[0]), float(grid[-1])), (0.02, 1.0), xlog=True, ylog=True)
        s.append(f'<text x="{x0 + 115}" y="26" text-anchor="middle" class="fig-bold">{titles[p]}</text>')
        s.append(frame(ax))
        for e in (-12, -9, -6, -3):
            xv = 2.0**e
            s.append(f'<line x1="{ax.px(xv):.1f}" y1="{ax.y0}" x2="{ax.px(xv):.1f}" y2="{ax.y1}" class="stroke-grid"/>'
                     f'<text x="{ax.px(xv):.1f}" y="{ax.y1 + 16}" text-anchor="middle" class="fig-small">2^{e}</text>')
        for yv, lab in [(0.03, "0.03"), (0.1, "0.1"), (0.3, "0.3"), (1.0, "1")]:
            s.append(f'<line x1="{ax.x0}" y1="{ax.py(yv):.1f}" x2="{ax.x1}" y2="{ax.py(yv):.1f}" class="stroke-grid"/>')
            if p == 0:
                s.append(f'<text x="{ax.x0 - 6}" y="{ax.py(yv) + 4:.1f}" text-anchor="end" class="fig-small">{lab}</text>')
        for k, w in enumerate(widths):
            ys = np.clip(curves[p][k], 0.02, 1.0)
            s.append(f'<polyline points="{pts(ax, grid, ys)}" fill="none" class="{classes[k]}" stroke-width="2"/>')
            j = int(np.argmin(curves[p][k]))
            s.append(f'<circle cx="{ax.px(grid[j]):.1f}" cy="{ax.py(ys[j]):.1f}" r="5" class="{fills[k]}"/>')
        s.append(f'<text x="{x0 + 115}" y="256" text-anchor="middle" class="fig-small">learning rate</text>')
    for k, w in enumerate(widths):
        s.append(f'<line x1="{200 + 90 * k}" y1="278" x2="{220 + 90 * k}" y2="278" class="{classes[k]}" stroke-width="2"/>'
                 f'<text x="{226 + 90 * k}" y="282" class="fig-small">width {w}</text>')
    s.append('<text x="8" y="40" class="fig-small">final loss</text>')
    s.append("</svg>")
    return "".join(s)


def fig3(sigma, one, five_q, five_c):
    ax = Axes(60, 540, 40, 210, (0.0, 1.0), (0.0, 1.4))
    s = ['<svg viewBox="0 0 600 270" role="img" aria-label="Singular value out against singular value in for one quintic Newton-Schulz step, five quintic steps, and five cubic steps. Five quintic steps put every input above 0.01 inside a band near 1.">']
    s.append(frame(ax))
    s.append(f'<rect x="{ax.x0}" y="{ax.py(1.3):.1f}" width="{ax.x1 - ax.x0}" height="{ax.py(0.7) - ax.py(1.3):.1f}" class="fill-soft-c"/>')
    for xv in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{ax.px(xv):.1f}" y1="{ax.y0}" x2="{ax.px(xv):.1f}" y2="{ax.y1}" class="stroke-grid"/>'
                 f'<text x="{ax.px(xv):.1f}" y="{ax.y1 + 16}" text-anchor="middle" class="fig-small">{xv:g}</text>')
    for yv in (0.0, 0.5, 1.0):
        s.append(f'<line x1="{ax.x0}" y1="{ax.py(yv):.1f}" x2="{ax.x1}" y2="{ax.py(yv):.1f}" class="stroke-grid"/>'
                 f'<text x="{ax.x0 - 6}" y="{ax.py(yv) + 4:.1f}" text-anchor="end" class="fig-small">{yv:g}</text>')
    s.append(f'<polyline points="{pts(ax, sigma, one)}" fill="none" class="stroke-muted dashed" stroke-width="2"/>')
    s.append(f'<polyline points="{pts(ax, sigma, five_c)}" fill="none" class="stroke-a" stroke-width="2"/>')
    s.append(f'<polyline points="{pts(ax, sigma, five_q)}" fill="none" class="stroke-b" stroke-width="2"/>')
    legend = [("stroke-b", "tuned quintic, 5 steps"), ("stroke-a", "cubic, 5 steps"), ("stroke-muted dashed", "tuned quintic, 1 step")]
    for k, (cls, label) in enumerate(legend):
        y = ax.py(0.42) + 20 * k
        s.append(f'<line x1="{ax.px(0.55):.1f}" y1="{y:.1f}" x2="{ax.px(0.55) + 28:.1f}" y2="{y:.1f}" class="{cls}" stroke-width="2"/>'
                 f'<text x="{ax.px(0.55) + 36:.1f}" y="{y + 4:.1f}" class="fig-small">{label}</text>')
    s.append('<text x="300" y="252" text-anchor="middle" class="fig-small">singular value going in (after dividing by the Frobenius norm)</text>')
    s.append('<text x="12" y="30" class="fig-small">singular value coming out</text>')
    s.append("</svg>")
    return "".join(s)


def main() -> None:
    out_dir = None
    if "--svg" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--svg") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- hand example: one example, sign update, a hidden layer of n units
    print("hand example: phi = (1,0,2,1) repeated, eta = 0.001")
    for n in (4, 16):
        phi = np.tile([1.0, 0.0, 2.0, 1.0], n // 4)
        eta = 0.001
        sp_move = eta * np.abs(phi).sum()
        mup_move = (eta * 4.0 / n) * np.abs(phi).sum()
        W = eta * np.outer(np.ones(n), np.sign(phi))
        print(f"  n={n}: sum|phi|={np.abs(phi).sum():.0f}, SP change per unit={sp_move:.4f}, "
              f"muP (lr x 4/n) change per unit={mup_move:.4f}, spectral norm of SP update={np.linalg.norm(W, 2):.4f}")

    # ---- figure 1
    widths = [32, 64, 128, 256, 512]
    sp = [float(np.mean([one_step_size(n, False, s) for s in range(8)])) for n in widths]
    mu = [float(np.mean([one_step_size(n, True, s) for s in range(8)])) for n in widths]
    print("\nfigure 1, mean over 8 seeds of the RMS change in the 2nd hidden pre-activation")
    for n, a_, b_ in zip(widths, sp, mu):
        print(f"  width {n:4d}  SP {a_:.5f}  muP {b_:.5f}")
    print(f"  ratio 512/32  SP {sp[-1] / sp[0]:.2f}  muP {mu[-1] / mu[0]:.2f}")

    # ---- spectral view
    print("\nspectral norm of one Adam step on a rank-1 gradient (sign matrix times lr)")
    for n in (32, 128, 512):
        rng = np.random.default_rng(0)
        g = np.outer(rng.normal(size=n), np.abs(rng.normal(size=n)))
        step = np.sign(g)
        print(f"  n={n:3d}  lr 1e-3: {1e-3 * np.linalg.norm(step, 2):.4f}   lr 1e-3*32/n: {1e-3 * 32 / n * np.linalg.norm(step, 2):.4f}")

    # ---- figure 2
    grid = 2.0 ** np.arange(-12, -1)
    curves = []
    print("\nfigure 2, mean final loss over 4 seeds, 200 Adam steps, lr = 2^-12 .. 2^-2")
    for mup in (False, True):
        block = []
        for n in (32, 128, 512):
            ls = np.mean([[train_loss(n, mup, lr, s) for lr in grid] for s in range(4)], axis=0)
            ls = np.where(np.isfinite(ls), ls, 9.0)
            block.append(ls)
            print(f"  {'muP' if mup else 'SP '} width {n:3d} best lr 2^{int(np.log2(grid[np.argmin(ls)]))}  "
                  f"loss {ls.min():.4f}  " + " ".join(f"{v:.3f}" for v in ls))
        curves.append(block)

    # ---- Newton-Schulz
    print("\nhand example, cubic step on the singular values of G = diag(3, 1)")
    s0 = np.array([3.0, 1.0]) / np.sqrt(10.0)
    print(f"  after dividing by the Frobenius norm {np.sqrt(10):.4f}: {np.round(s0, 4)}")
    s = s0.copy()
    for k in range(1, 4):
        s = ns_scalar(s, 1, CUBIC)
        print(f"  cubic step {k}: {np.round(s, 4)}")
    s = s0.copy()
    for k in range(1, 6):
        s = ns_scalar(s, 1, QUINTIC)
        print(f"  quintic step {k}: {np.round(s, 4)}")
    print("  without the division, sigma=3 under the cubic: " +
          ", ".join(f"{v:.1f}" for v in [3.0, ns_scalar(3.0, 1, CUBIC), ns_scalar(3.0, 2, CUBIC), ns_scalar(3.0, 3, CUBIC)]))
    print(f"  quintic a+b+c = {A + B + C:.4f}")

    sigma = np.linspace(0.0, 1.0, 101)
    one = ns_scalar(sigma, 1, QUINTIC)
    five_q = ns_scalar(sigma, 5, QUINTIC)
    five_c = ns_scalar(sigma, 5, CUBIC)
    for lo in (0.01, 0.05, 0.1):
        m = sigma >= lo
        print(f"  quintic 5 steps, sigma in [{lo}, 1]: out in [{five_q[m].min():.3f}, {five_q[m].max():.3f}]")
    print(f"  quintic 5 steps at sigma 0.01: {ns_scalar(0.01, 5, QUINTIC):.3f}, at 0.001: {ns_scalar(0.001, 5, QUINTIC):.4f}")
    print(f"  cubic 5 steps at sigma 0.05: {ns_scalar(0.05, 5, CUBIC):.3f}, at 0.2: {ns_scalar(0.2, 5, CUBIC):.3f}")

    # a real matrix, to quote the band
    rng = np.random.default_rng(0)
    G = rng.normal(size=(64, 32))
    X = G / np.linalg.norm(G)
    print(f"\n64x32 Gaussian G: singular values before {np.linalg.svd(G, compute_uv=False)[[0, -1]].round(2)}, "
          f"after normalising {np.linalg.svd(X, compute_uv=False)[[0, -1]].round(3)}")
    for _ in range(5):
        S = X @ X.T
        X = A * X + B * (S @ X) + C * (S @ S @ X)
    sv = np.linalg.svd(X, compute_uv=False)
    print(f"  after 5 quintic steps: min {sv.min():.3f} max {sv.max():.3f}")

    if out_dir is not None:
        (out_dir / "fig1.svg").write_text(fig1(sp, mu, widths))
        (out_dir / "fig2.svg").write_text(fig2(curves, grid, [32, 128, 512]))
        (out_dir / "fig3.svg").write_text(fig3(sigma, one, five_q, five_c))
        print(f"\nwrote SVGs to {out_dir}")


if __name__ == "__main__":
    main()
