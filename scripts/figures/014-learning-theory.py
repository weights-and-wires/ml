#!/usr/bin/env python3
"""Numbers and SVG for the learning-theory figures.

Figure 1: Hoeffding's bound against the simulated chance that the average of n
fair-coin flips misses 0.5 by more than 0.1.
Figure 2: perceptron mistakes against the (R / margin)^2 bound on 40 seeded
separable problems with different margins.
Figure 3: median test error of min-norm least squares on p random ReLU
features, with n = 30 training points.

Run it to print every value the lesson quotes. Run it with `--svg` to also
print the three SVG bodies that the lesson uses.
"""

from __future__ import annotations

import math
import sys

import numpy as np


# ------------------------------------------------------------------ the maths


def hoeffding(n: int, eps: float) -> float:
    return min(1.0, 2.0 * math.exp(-2.0 * n * eps**2))


def deviation_frequency(n: int, eps: float, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean(np.abs(rng.binomial(n, 0.5, size=trials) / n - 0.5) > eps))


def sample_size(k: int, eps: float, delta: float) -> int:
    return int(math.ceil(math.log(2.0 * k / delta) / (2.0 * eps**2)))


def perceptron(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    w = np.zeros(X.shape[1])
    mistakes = 0
    for _ in range(10000):
        clean = True
        for x_i, y_i in zip(X, y):
            if y_i * (w @ x_i) <= 0:
                w = w + y_i * x_i
                mistakes += 1
                clean = False
        if clean:
            break
    return w, mistakes


def perceptron_bound(X: np.ndarray, y: np.ndarray, w_star: np.ndarray) -> float:
    R = np.max(np.linalg.norm(X, axis=1))
    gamma = np.min(y * (X @ w_star)) / np.linalg.norm(w_star)
    return float((R / gamma) ** 2)


def min_norm_test_error(n: int, p: int, trials: int, seed: int, d: int = 3, noise: float = 0.3) -> float:
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(trials):
        beta = rng.normal(size=d)
        X = rng.normal(size=(n, d))
        X_test = rng.normal(size=(400, d))
        W = rng.normal(size=(d, p))
        b = rng.normal(size=p)
        y = np.sin(X @ beta) + noise * rng.normal(size=n)
        F = np.maximum(X @ W + b, 0.0)
        F_test = np.maximum(X_test @ W + b, 0.0)
        w = np.linalg.pinv(F) @ y
        errors.append(np.mean((F_test @ w - np.sin(X_test @ beta)) ** 2) + noise**2)
    return float(np.median(errors))


# --------------------------------------------------------------- SVG helpers


def axes(x0, y0, x1, y1):
    return (
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="stroke-muted"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="stroke-muted"/>'
    )


def poly(points, cls):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" class="{cls}" stroke-width="2.5"/>'


def dots(points, cls, r=4):
    return "\n".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}"/>' for x, y in points)


# -------------------------------------------------------------------- figures


def figure1(svg: bool):
    ns = list(range(10, 160, 10))
    eps = 0.1
    bound = [hoeffding(n, eps) for n in ns]
    freq = [deviation_frequency(n, eps, 400000, 0) for n in ns]
    print("Figure 1: n, Hoeffding bound, simulated frequency (eps = 0.1, 400000 trials)")
    for n, b, f in zip(ns, bound, freq):
        print(f"  {n:4d}  {b:.5f}  {f:.5f}  ratio {b / f if f else float('inf'):.1f}")
    if not svg:
        return
    x0, x1, yt, yb = 60, 600, 40, 230
    lo, hi = -5.0, 0.0

    def px(n):
        return x0 + (n - 10) / 140.0 * (x1 - x0)

    def py(v):
        return yb - (math.log10(v) - lo) / (hi - lo) * (yb - yt)

    out = ['<svg viewBox="0 0 640 290" role="img" aria-label="Hoeffding bound and simulated deviation chance against n on a log axis; both fall in parallel lines">']
    for e in range(-5, 1):
        y = py(10.0**e)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        label = "1" if e == 0 else f"1e{e}"
        out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" class="fig-small">{label}</text>')
    for n in (10, 50, 100, 150):
        out.append(f'<text x="{px(n):.1f}" y="{yb + 16}" text-anchor="middle" class="fig-small">{n}</text>')
    out.append(axes(x0, yt, x1, yb))
    out.append(poly([(px(n), py(b)) for n, b in zip(ns, bound)], "stroke-b"))
    out.append(dots([(px(n), py(b)) for n, b in zip(ns, bound)], "fill-b"))
    pts = [(px(n), py(f)) for n, f in zip(ns, freq) if f > 0]
    out.append(poly(pts, "stroke-a"))
    out.append(dots(pts, "fill-a"))
    out.append(f'<text x="{px(100) + 10:.1f}" y="{py(hoeffding(100, eps)) - 10:.1f}" class="fig-bold">bound 2 exp(-2 n eps²)</text>')
    out.append(f'<text x="{px(100) + 10:.1f}" y="{py(freq[9]) + 26:.1f}" class="fig-bold">simulated chance</text>')
    out.append(f'<text x="330" y="{yb + 36}" text-anchor="middle">n, coin flips per average</text>')
    out.append(f'<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">chance of a miss above 0.1</text>')
    out.append("</svg>")
    print("\n".join(out))


def figure2(svg: bool):
    rng = np.random.default_rng(0)
    rows = []
    w_star = np.array([1.0, 0.0])
    for margin in np.repeat(np.array([1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008]), 5):
        X = rng.uniform(-3, 3, size=(300, 2))
        X = X[np.abs(X @ w_star) >= margin][:100]
        y = np.sign(X @ w_star)
        w, m = perceptron(X, y)
        rows.append((perceptron_bound(X, y, w_star), m))
    print("Figure 2: bound, mistakes (40 problems; margins 1 down to 0.008)")
    for b, m in rows[::5]:
        print(f"  bound {b:10.1f}  mistakes {m:4d}")
    print("  all below bound:", all(m <= b for b, m in rows), " mean mistakes / bound:", round(float(np.mean([m / b for b, m in rows])), 3))
    if not svg:
        return
    x0, x1, yt, yb = 70, 590, 30, 230
    lo, hi = 0.0, 5.0  # log10 range 1 .. 100000

    def px(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    def py(v):
        return yb - (math.log10(v) - lo) / (hi - lo) * (yb - yt)

    out = ['<svg viewBox="0 0 640 290" role="img" aria-label="Perceptron mistakes against the bound, both on log axes: every point sits below the line where mistakes equal the bound">']
    for e in range(0, 6):
        v = 10.0**e
        out.append(f'<line x1="{px(v):.1f}" y1="{yt}" x2="{px(v):.1f}" y2="{yb}" class="stroke-grid"/>')
        out.append(f'<line x1="{x0}" y1="{py(v):.1f}" x2="{x1}" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{px(v):.1f}" y="{yb + 16}" text-anchor="middle" class="fig-small">{int(v)}</text>')
        out.append(f'<text x="{x0 - 6}" y="{py(v) + 4:.1f}" text-anchor="end" class="fig-small">{int(v)}</text>')
    out.append(axes(x0, yt, x1, yb))
    out.append(f'<line x1="{px(1):.1f}" y1="{py(1):.1f}" x2="{px(1e5):.1f}" y2="{py(1e5):.1f}" class="stroke-b dashed" stroke-width="2"/>')
    out.append(dots([(px(max(b, 1)), py(max(m, 1))) for b, m in rows], "fill-a"))
    out.append(f'<text x="{px(2):.1f}" y="{py(1500):.1f}" class="fig-bold">mistakes = bound</text>')
    out.append(f'<text x="330" y="{yb + 36}" text-anchor="middle">bound (R / margin)²</text>')
    out.append(f'<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">mistakes made</text>')
    out.append("</svg>")
    print("\n".join(out))


def figure3(svg: bool):
    n = 30
    ps = [5, 10, 15, 20, 25, 28, 30, 32, 35, 40, 50, 60, 90, 150, 300, 600]
    errs = [min_norm_test_error(n, p, 60, 0) for p in ps]
    print("Figure 3: p, median test error (n = 30, 60 trials)")
    for p, e in zip(ps, errs):
        print(f"  {p:4d}  {e:.3f}")
    if not svg:
        return
    x0, x1, yt, yb = 60, 600, 40, 230
    lo, hi = -1.0, 2.0  # 0.1 .. 100

    def px(p):
        return x0 + (math.log10(p) - math.log10(5)) / (math.log10(600) - math.log10(5)) * (x1 - x0)

    def py(v):
        return yb - (math.log10(v) - lo) / (hi - lo) * (yb - yt)

    out = ['<svg viewBox="0 0 640 290" role="img" aria-label="Test error against the number of features p with 30 training points: low, then a spike at p = 30, then falling again">']
    for e, label in ((-1, "0.1"), (0, "1"), (1, "10"), (2, "100")):
        y = py(10.0**e)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" class="fig-small">{label}</text>')
    for p in (5, 10, 30, 100, 300, 600):
        out.append(f'<text x="{px(p):.1f}" y="{yb + 16}" text-anchor="middle" class="fig-small">{p}</text>')
    out.append(axes(x0, yt, x1, yb))
    out.append(f'<line x1="{px(n):.1f}" y1="{yt}" x2="{px(n):.1f}" y2="{yb}" class="stroke-muted dashed"/>')
    pts = [(px(p), py(e)) for p, e in zip(ps, errs)]
    out.append(poly(pts, "stroke-b"))
    out.append(dots(pts, "fill-b"))
    out.append(f'<text x="{px(n) + 8:.1f}" y="{yt + 12}" class="fig-bold">p = n = 30</text>')
    out.append(f'<text x="330" y="{yb + 36}" text-anchor="middle">p, number of random features (log axis)</text>')
    out.append(f'<text x="14" y="135" transform="rotate(-90 14 135)" text-anchor="middle">median test error</text>')
    out.append("</svg>")
    print("\n".join(out))


def numbers():
    print("Hoeffding n=100 eps=0.1:", round(hoeffding(100, 0.1), 4))
    print("Hoeffding n=100 eps=0.1 wrong (no 2, no 2n):", round(math.exp(-100 * 0.01), 4))
    print("Sample sizes at eps=0.05, delta=0.05:")
    for k in (1, 10, 1000, 10**6):
        print(f"  k={k:>8d}  n={sample_size(k, 0.05, 0.05)}")
    X = np.array([[1.0, 1.0], [1.0, -2.0], [-1.0, -2.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, -1.0, 1.0])
    w, m = perceptron(X, y)
    print("Perceptron hand example: w =", w, "mistakes =", m, "bound =", perceptron_bound(X, y, np.array([1.0, 0.0])))


if __name__ == "__main__":
    flag = "--svg" in sys.argv
    numbers()
    figure1(flag)
    figure2(flag)
    figure3(flag)
