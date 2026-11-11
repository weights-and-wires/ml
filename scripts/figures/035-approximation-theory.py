#!/usr/bin/env python3
"""Numbers and SVG for the approximation-theory figures.

Figure 1: sin on [0, pi], its 4-segment piecewise-linear interpolant, and the
largest gap, against the h^2/8 bound.
Figure 2: largest interpolation error of sin against the number of segments,
on log-log axes, with the fitted slope.
Figure 3: hidden units needed for a sawtooth with 2^k pieces, by a deep
network (2k units) and a one-hidden-layer network (2^k - 1 units).

Run it to print every value the lesson quotes. Run it with `--svg` to also
print the three SVG bodies that the lesson uses.
"""

from __future__ import annotations

import math
import sys

import numpy as np


def interp_error(f, n_segments: int, a: float, b: float, fine: int = 20001) -> float:
    knots = np.linspace(a, b, n_segments + 1)
    xs = np.linspace(a, b, fine)
    return float(np.max(np.abs(f(xs) - np.interp(xs, knots, f(knots)))))


def relu(x):
    return np.maximum(x, 0.0)


def tent(x):
    return 2.0 * relu(x) - 4.0 * relu(x - 0.5)


def sawtooth(x, k):
    for _ in range(k):
        x = tent(x)
    return x


def count_pieces(f, xs, tol=1e-6):
    slopes = np.diff(f(xs)) / np.diff(xs)
    return 1 + int(np.sum(np.abs(np.diff(slopes)) > tol))


def main(svg: bool) -> None:
    a, b = 0.0, math.pi
    print("--- error of the sin interpolant, M = max|g''| = 1")
    segs = [2, 4, 8, 16, 32, 64, 128, 256]
    errs = []
    for n in segs:
        h = (b - a) / n
        e = interp_error(np.sin, n, a, b)
        errs.append(e)
        print(f"segments {n:4d}  h {h:.5f}  error {e:.6f}  bound h^2/8 {h*h/8:.6f}")
    for i in range(1, len(segs)):
        print(f"  ratio error({segs[i-1]})/error({segs[i]}) = {errs[i-1]/errs[i]:.3f}")
    slope = np.polyfit(np.log(segs), np.log(errs), 1)[0]
    slope_raw = np.polyfit(segs, errs, 1)[0]
    print(f"log-log slope {slope:.4f}   unlogged slope {slope_raw:.6f}")

    print("--- x^2 on [0, 1], 2 segments")
    print("chord at 0.25:", 0.5 * 0.25, " true:", 0.25**2, " gap:", 0.125 - 0.0625)

    print("--- segments needed for error eps, sin on [0, pi]")
    for eps in [1e-2, 1e-3, 1e-4]:
        need = math.ceil(math.pi * math.sqrt(1.0 / (8 * eps)))
        got = interp_error(np.sin, need, a, b)
        print(f"eps {eps:g}: formula {need}  measured error with that many {got:.6f}")

    print("--- grid cells in d dimensions, curvature 8, eps 0.01 (10 per axis)")
    for d in [1, 2, 3, 5, 10]:
        print(f"d {d:2d}: {10**d:,} cells")

    print("--- sawtooth")
    xs = np.linspace(0, 1, 2**12 + 1)
    for k in range(1, 9):
        print(f"k {k}: pieces {count_pieces(lambda x: sawtooth(x, k), xs)}  min {sawtooth(xs, k).min():.3f}  max {sawtooth(xs, k).max():.3f}")
    print("T(T(x)) on 9 points:", sawtooth(np.linspace(0, 1, 9), 2))
    print("--- units")
    for k in [1, 3, 4, 6, 10, 20]:
        print(f"k {k:2d}: deep {2*k}  shallow {2**k - 1:,}")

    if not svg:
        return
    # Figure 1
    n1 = 4
    X0, X1, Y0, Y1 = 60.0, 560.0, 190.0, 40.0  # x range, y base, y top (value 1)
    px = lambda x: X0 + (x - a) / (b - a) * (X1 - X0)
    py = lambda y: Y0 + y * (Y1 - Y0)
    xs = np.linspace(a, b, 120)
    curve = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(np.sin(x)):.1f}" for i, x in enumerate(xs))
    knots = np.linspace(a, b, n1 + 1)
    chord = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(np.sin(x)):.1f}" for i, x in enumerate(knots))
    print("\n<!-- figure 1 -->")
    print(f'<path d="{curve}" fill="none" class="stroke-a" stroke-width="2.4"/>')
    print(f'<path d="{chord}" fill="none" class="stroke-b" stroke-width="2.4"/>')
    for x in knots:
        print(f'<circle cx="{px(x):.1f}" cy="{py(np.sin(x)):.1f}" r="3.5" class="fill-ink"/>')
    gap_x = math.pi / 2
    # largest gap is at the middle of the segment containing pi/2 ... segment 2 spans [pi/4, pi/2]; mid is 3pi/8
    xm = 3 * math.pi / 8
    ym = 0.5 * (math.sin(math.pi / 4) + 1.0)
    print(f'<line x1="{px(xm):.1f}" y1="{py(np.sin(xm)):.1f}" x2="{px(xm):.1f}" y2="{py(ym):.1f}" class="stroke-bad" stroke-width="2.4"/>')
    print(f"gap at 3pi/8: {ym - np.sin(xm):.4f}; max over grid {interp_error(np.sin, 4, a, b):.4f}; bound {(b-a)**2/16/8:.4f}")
    # Figure 2
    L0, L1, T0, T1 = 70.0, 560.0, 30.0, 190.0
    lx = lambda n: L0 + (math.log2(n) - 1) / 7 * (L1 - L0)
    le = [math.log10(e) for e in errs]
    lo, hi = -5.0, -0.0
    ly = lambda e: T1 + (math.log10(e) - lo) / (hi - lo) * (T0 - T1)
    print("\n<!-- figure 2 -->")
    pts = " ".join(f"{'M' if i == 0 else 'L'}{lx(n):.1f},{ly(e):.1f}" for i, (n, e) in enumerate(zip(segs, errs)))
    print(f'<path d="{pts}" fill="none" class="stroke-b" stroke-width="2"/>')
    for n, e in zip(segs, errs):
        print(f'<circle cx="{lx(n):.1f}" cy="{ly(e):.1f}" r="3.5" class="fill-b"/>  <!-- n={n} err={e:.5f} -->')
    # reference slope -2 through first point
    r0 = errs[0]
    print(f'<line x1="{lx(2):.1f}" y1="{ly(r0):.1f}" x2="{lx(256):.1f}" y2="{ly(r0 * (2/256)**2):.1f}" class="stroke-muted dashed"/>')
    for n in segs:
        print(f'<text x="{lx(n):.1f}" y="{T1+16:.0f}" text-anchor="middle" class="fig-small">{n}</text>')
    for p in range(-5, 1):
        print(f'<text x="{L0-8:.0f}" y="{ly(10.0**p)+4:.1f}" text-anchor="end" class="fig-small">1e{p}</text>' if p else f'<text x="{L0-8:.0f}" y="{ly(1.0)+4:.1f}" text-anchor="end" class="fig-small">1</text>')
    # Figure 3
    ks = list(range(1, 13))
    M0, M1, U0, U1 = 60.0, 560.0, 190.0, 30.0
    kx = lambda k: M0 + (k - 1) / 11 * (M1 - M0)
    top = math.log10(4096)
    uy = lambda u: U0 + math.log10(u) / top * (U1 - U0)
    print("\n<!-- figure 3 -->")
    deep = " ".join(f"{'M' if i == 0 else 'L'}{kx(k):.1f},{uy(2*k):.1f}" for i, k in enumerate(ks))
    shallow = " ".join(f"{'M' if i == 0 else 'L'}{kx(k):.1f},{uy(2**k - 1):.1f}" for i, k in enumerate(ks))
    print(f'<path d="{deep}" fill="none" class="stroke-a" stroke-width="2.4"/>')
    print(f'<path d="{shallow}" fill="none" class="stroke-b" stroke-width="2.4"/>')
    for k in ks:
        print(f'<circle cx="{kx(k):.1f}" cy="{uy(2*k):.1f}" r="3" class="fill-a"/><circle cx="{kx(k):.1f}" cy="{uy(2**k-1):.1f}" r="3" class="fill-b"/>')
        print(f'<text x="{kx(k):.1f}" y="{U0+16:.0f}" text-anchor="middle" class="fig-small">{k}</text>')
    for u in [1, 10, 100, 1000]:
        print(f'<text x="{M0-8:.0f}" y="{uy(u)+4:.1f}" text-anchor="end" class="fig-small">{u}</text>')


if __name__ == "__main__":
    main("--svg" in sys.argv)
