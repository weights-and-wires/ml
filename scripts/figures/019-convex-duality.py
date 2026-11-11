#!/usr/bin/env python3
"""Numbers and SVG for the convex duality chapter.

Prints every value the chapter quotes: the chord test, the one-variable dual
function, the small LP, the equality-constrained quadratic, the ridge dual, and
the lasso duality gap, then the gap along a FISTA run. Pass `--svg DIR` to also
write the three figure bodies as `fig1.svg`, `fig2.svg`, and `fig3.svg`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------- numbers


def one_variable() -> None:
    print("== one variable: minimise x^2 subject to x >= 1 ==")
    for lam in [0.0, 1.0, 2.0, 3.0, 4.0]:
        x = lam / 2.0
        g = x**2 + lam * (1 - x)
        print(f"lambda={lam}: argmin x={x}, g(lambda)={g:.4f} (formula {lam - lam**2 / 4:.4f})")


def chord() -> None:
    print("== chord test ==")
    f = lambda x: x**2
    print("convex x^2: f(0.5 * (-1 + 2))=", f(0.5), " average of ends=", 0.5 * (f(-1) + f(2)))
    h = lambda x: x**4 - 2 * x**2 + 0.5 * x
    xs = np.linspace(-1.6, 1.6, 32001)
    v = h(xs)
    mins = [i for i in range(1, len(xs) - 1) if v[i] < v[i - 1] and v[i] < v[i + 1]]
    for i in mins:
        print(f"double well local min at x={xs[i]:.3f}, h={v[i]:.4f}")
    print("double well chord check: h(0)=", h(0.0), " average of h(-1.2), h(1.2)=", 0.5 * (h(-1.2) + h(1.2)))


def lp() -> None:
    print("== LP: min 2 x1 + 3 x2, x1 + x2 >= 4, x1 + 3 x2 >= 6, x >= 0 ==")
    c = np.array([2.0, 3.0])
    A = np.array([[1.0, 1.0], [1.0, 3.0]])
    b = np.array([4.0, 6.0])
    p = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, None)] * 2)
    d = linprog(-b, A_ub=A.T, b_ub=c, bounds=[(0, None)] * 2)
    print("primal x*", p.x, "value", p.fun)
    print("dual y*", d.x, "value", -d.fun)


def qp() -> None:
    print("== QP: min 0.5 |x|^2 s.t. x1 + x2 = 1 ==")
    for nu in [-1.5, -1.0, -0.5, 0.0]:
        print(f"nu={nu}: g(nu)={-nu**2 - nu:.4f}")


def ridge() -> None:
    print("== ridge: X=[[1],[2]], y=[1,2], lambda=1 ==")
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    K = X @ X.T
    a = np.linalg.solve(K + np.eye(2), y)
    w = X.T @ a
    P = 0.5 * np.sum((y - X @ w) ** 2) + 0.5 * float(w @ w)
    print("alpha", a, "w", w, "primal", P, "closed-form w", 5 / 6)


def lasso_value(X, y, w, lam):
    return 0.5 * np.sum((X @ w - y) ** 2) + lam * np.sum(np.abs(w))


def lasso_gap(X, y, w, lam, rescale=True):
    r = y - X @ w
    s = min(1.0, lam / max(np.max(np.abs(X.T @ r)), 1e-300)) if rescale else 1.0
    theta = s * r
    dual = 0.5 * y @ y - 0.5 * np.sum((y - theta) ** 2)
    return lasso_value(X, y, w, lam) - dual


def lasso_tiny() -> None:
    print("== lasso: X=[[1],[2]], y=[1,2], lambda=1 ==")
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    for wv in [0.0, 0.4, 0.8]:
        w = np.array([wv])
        print(
            f"w={wv}: primal={lasso_value(X, y, w, 1.0):.4f} gap={lasso_gap(X, y, w, 1.0):.4f}"
            f" unscaled-gap={lasso_gap(X, y, w, 1.0, rescale=False):.4f}"
        )


def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def lasso_data():
    """100 houses, 20 features, 4 of which matter."""
    rng = np.random.default_rng(0)
    n, d = 100, 20
    X = rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[:4] = [3.0, -2.0, 1.5, 2.0]
    y = X @ w_true + 0.5 * rng.normal(size=n)
    return X, y


def fista_run(X, y, lam, iters):
    step = 1.0 / np.linalg.eigvalsh(X.T @ X).max()
    w = np.zeros(X.shape[1])
    z = w.copy()
    t = 1.0
    path = [w.copy()]
    for _ in range(iters):
        w_new = soft(z - step * X.T @ (X @ z - y), step * lam)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = w_new + (t - 1) / t_new * (w_new - w)
        w, t = w_new, t_new
        path.append(w.copy())
    return path


def fista_gaps(iters=100, lam=10.0):
    X, y = lasso_data()
    path = fista_run(X, y, lam, 3000)
    p_star = lasso_value(X, y, path[-1], lam)
    rows = []
    for k in range(iters + 1):
        rows.append((k, lasso_value(X, y, path[k], lam) - p_star, lasso_gap(X, y, path[k], lam)))
    return rows, p_star


def lasso_fista() -> None:
    print("== lasso FISTA run, n=100, d=20, lambda=10 ==")
    rows, p_star = fista_gaps()
    print("p* =", p_star)
    for k, sub, gap in rows:
        if k in (0, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100):
            print(f"iter {k}: suboptimality={max(sub, 0):.3e} gap={gap:.3e}")
    first = next((k for k, _, gap in rows if gap < 1e-3), None)
    print("first iteration with gap < 1e-3:", first)


# ------------------------------------------------------------------ svg


def poly(pts, cls, extra=""):
    s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{s}" fill="none" class="{cls}" stroke-width="2"{extra}/>'


def fig1() -> str:
    h = lambda x: x**4 - 2 * x**2 + 0.5 * x
    f = lambda x: x**2

    def panel(ox, fn, lo, hi, ylo, yhi, title, label_pts, chord_pts):
        W, H, top = 230.0, 140.0, 50.0
        sx = lambda x: ox + (x - lo) / (hi - lo) * W
        sy = lambda y: top + H - (y - ylo) / (yhi - ylo) * H
        xs = np.linspace(lo, hi, 80)
        out = [
            f'<line x1="{ox}" y1="{top + H}" x2="{ox + W}" y2="{top + H}" class="stroke-muted"/>',
            f'<text x="{ox}" y="28" class="fig-bold">{title}</text>',
            poly([(sx(x), sy(fn(x))) for x in xs], "stroke-a"),
        ]
        (x1, x2) = chord_pts
        out.append(
            f'<line x1="{sx(x1):.1f}" y1="{sy(fn(x1)):.1f}" x2="{sx(x2):.1f}" y2="{sy(fn(x2)):.1f}" class="stroke-b dashed" stroke-width="2"/>'
        )
        for x in (x1, x2):
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(fn(x)):.1f}" r="4.5" class="fill-b"/>')
        for x, txt in label_pts:
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(fn(x)):.1f}" r="5" class="fill-c"/>')
            out.append(
                f'<text x="{sx(x):.1f}" y="{sy(fn(x)) + 20:.1f}" class="fig-small" text-anchor="middle">{txt}</text>'
            )
        return out

    xs = np.linspace(-1.6, 1.6, 32001)
    v = h(xs)
    mins = [xs[i] for i in range(1, len(xs) - 1) if v[i] < v[i - 1] and v[i] < v[i + 1]]
    left, right = mins[0], mins[1]
    parts = ['<svg viewBox="0 0 600 240" role="img" aria-label="Left: a bowl with its chord above it. Right: a curve with two dips, where the chord dips below the curve.">']
    parts += panel(30, f, -1.5, 1.5, -0.3, 2.4, "Convex: chord above the curve", [(0.0, f"min {f(0.0):.1f}")], (-1.2, 1.2))
    parts += panel(330, h, -1.6, 1.6, -2.0, 3.5, "Not convex: two dips", [(left, f"{h(left):.2f}"), (right, f"{h(right):.2f}")], (-1.2, 1.2))
    parts.append("</svg>")
    return "\n".join(parts)


def fig2() -> str:
    lams = [0.0, 1.0, 2.0, 3.0]
    cls = ["stroke-muted", "stroke-a", "stroke-b", "stroke-d"]
    fcls = ["fill-muted", "fill-a", "fill-b", "fill-d"]
    ox, top, W, H = 40.0, 40.0, 240.0, 170.0
    sx = lambda x: ox + (x + 0.5) / 3.5 * W
    sy = lambda y: top + H - (y + 0.5) / 5.5 * H
    parts = ['<svg viewBox="0 0 600 262" role="img" aria-label="Left: four curves of x squared plus lambda times one minus x, each with its lowest point below the primal optimum 1. Right: the dual function lambda minus lambda squared over 4, peaking at 1 when lambda is 2.">']
    parts.append(f'<line x1="{ox}" y1="{top + H}" x2="{ox + W}" y2="{top + H}" class="stroke-muted"/>')
    parts.append(f'<line x1="{ox}" y1="{sy(1):.1f}" x2="{ox + W}" y2="{sy(1):.1f}" class="stroke-ink dashed"/>')
    parts.append(f'<text x="{ox + W - 4}" y="{sy(1) - 6:.1f}" class="fig-small" text-anchor="end">p* = 1</text>')
    parts.append(f'<text x="{ox}" y="22" class="fig-bold">Lagrangian in x, for four prices</text>')
    xs = np.linspace(-0.5, 3.0, 70)
    for lam, c, fc in zip(lams, cls, fcls):
        parts.append(poly([(sx(x), sy(x * x + lam * (1 - x))) for x in xs if -0.5 <= x * x + lam * (1 - x) <= 5.0], c))
        xm = lam / 2.0
        gm = lam - lam**2 / 4
        parts.append(f'<circle cx="{sx(xm):.1f}" cy="{sy(gm):.1f}" r="4.5" class="{fc}"/>')
    # legend labels at the minima
    for lam, fc in zip(lams, fcls):
        xm, gm = lam / 2.0, lam - lam**2 / 4
        parts.append(f'<text x="{sx(xm):.1f}" y="{sy(gm) + 18:.1f}" class="fig-small" text-anchor="middle">{gm:.2f}</text>')
    parts.append(f'<text x="{ox + W / 2:.0f}" y="{top + H + 28}" class="fig-small" text-anchor="middle">x   (curves for price 0, 1, 2, 3)</text>')
    # right panel
    ox2, W2 = 340.0, 230.0
    sx2 = lambda l: ox2 + l / 4.0 * W2
    sy2 = lambda y: top + H - (y + 0.5) / 5.5 * H
    parts.append(f'<text x="{ox2}" y="22" class="fig-bold">Dual function g(price)</text>')
    parts.append(f'<line x1="{ox2}" y1="{top + H}" x2="{ox2 + W2}" y2="{top + H}" class="stroke-muted"/>')
    parts.append(f'<line x1="{ox2}" y1="{sy2(1):.1f}" x2="{ox2 + W2}" y2="{sy2(1):.1f}" class="stroke-ink dashed"/>')
    ls = np.linspace(0, 4, 60)
    parts.append(poly([(sx2(l), sy2(l - l * l / 4)) for l in ls], "stroke-c"))
    for lam, fc in zip(lams, fcls):
        parts.append(f'<circle cx="{sx2(lam):.1f}" cy="{sy2(lam - lam**2 / 4):.1f}" r="4.5" class="{fc}"/>')
    for l in [0, 1, 2, 3, 4]:
        parts.append(f'<text x="{sx2(l):.1f}" y="{top + H + 16}" class="fig-small" text-anchor="middle">{l}</text>')
    parts.append(f'<text x="{sx2(2):.1f}" y="{sy2(1) - 8:.1f}" class="fig-small" text-anchor="middle">gap 0 at price 2</text>')
    parts.append(f'<text x="{ox2 + W2 / 2:.0f}" y="{top + H + 32}" class="fig-small" text-anchor="middle">price lambda</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def fig3() -> str:
    rows, _ = fista_gaps(60)
    ox, top, W, H = 60.0, 40.0, 500.0, 170.0
    lo, hi = -6.0, 3.0
    sx = lambda k: ox + k / 60.0 * W
    sy = lambda v: top + H - (np.log10(max(v, 1e-6)) - lo) / (hi - lo) * H
    parts = ['<svg viewBox="0 0 600 262" role="img" aria-label="Duality gap and true suboptimality of the lasso objective over 60 FISTA steps on a log scale; the gap stays above the suboptimality and both fall.">']
    parts.append(f'<text x="{ox}" y="22" class="fig-bold">Lasso duality gap over 60 FISTA steps (log scale)</text>')
    for e in range(-6, 4, 3):
        y = sy(10.0**e)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox + W}" y2="{y:.1f}" class="stroke-grid"/>')
        parts.append(f'<text x="{ox - 6}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">1e{e}</text>')
    parts.append(f'<line x1="{ox}" y1="{top + H}" x2="{ox + W}" y2="{top + H}" class="stroke-muted"/>')
    for k in range(0, 61, 10):
        parts.append(f'<text x="{sx(k):.1f}" y="{top + H + 16}" class="fig-small" text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{ox + W / 2:.0f}" y="{top + H + 34}" class="fig-small" text-anchor="middle">FISTA step</text>')
    parts.append(poly([(sx(k), sy(g)) for k, _, g in rows], "stroke-b"))
    parts.append(poly([(sx(k), sy(max(s, 1e-6))) for k, s, _ in rows], "stroke-a"))
    parts.append(f'<text x="{sx(34):.1f}" y="{sy(rows[34][2]) - 8:.1f}" class="fig-small">gap (computable)</text>')
    parts.append(f'<text x="{sx(44):.1f}" y="{sy(max(rows[44][1], 1e-6)) + 18:.1f}" class="fig-small">true suboptimality (unknown in practice)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    chord()
    one_variable()
    lp()
    qp()
    ridge()
    lasso_tiny()
    lasso_fista()
    if "--svg" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--svg") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3)):
            (out / f"{name}.svg").write_text(fn())
        print("wrote svg to", out)
