#!/usr/bin/env python3
"""Numbers for the GAN and generator-evaluation lesson.

Figure 1 plots two 1-D bells, real p = N(0, 1) and generated q = N(2, 1), and
the optimal discriminator D*(x) = p(x) / (p(x) + q(x)) = sigmoid(2 - 2x).
Figure 2 plots the size of the generator's gradient with respect to the
discriminator logit b on a fake sample, for the saturating loss log(1 - D)
and the non-saturating loss -log D.
Figure 3 plots a two-bell real density, a collapsed generator on one bell, and
a single Gaussian with the same mean and variance as the mixture.

Prints the hand-check values, then the SVG point lists that the lesson draws.
"""

from __future__ import annotations

import math

import numpy as np


def normal(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def js(p, q, dx):
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / m), 0.0)
        b = np.where(q > 0, q * np.log(q / m), 0.0)
    return 0.5 * (a.sum() + b.sum()) * dx


def fid_1d(m1, s1, m2, s2):
    return (m1 - m2) ** 2 + (s1 - s2) ** 2


def knn_radius(x, k):
    d = np.abs(x[:, None] - x[None, :])
    return np.sort(d, axis=1)[:, k]


def precision_recall(real, fake, k=3):
    rr, rf = knn_radius(real, k), knn_radius(fake, k)
    d = np.abs(fake[:, None] - real[None, :])
    precision = float(np.mean(np.any(d <= rr[None, :], axis=1)))
    recall = float(np.mean(np.any(d.T <= rf[None, :], axis=1)))
    return precision, recall


def main() -> None:
    # Figure 1.
    x = np.linspace(-4.0, 6.0, 2001)
    dx = x[1] - x[0]
    p, q = normal(x, 0.0, 1.0), normal(x, 2.0, 1.0)
    dstar = p / (p + q)
    print("p(0), q(0), D*(0):", normal(0.0, 0, 1), normal(0.0, 2, 1), dstar[np.argmin(abs(x))])
    print("D*(1) (equal densities):", p[np.argmin(abs(x - 1))] / (p + q)[np.argmin(abs(x - 1))])
    print("max |D* - sigmoid(2 - 2x)|:", np.max(np.abs(dstar - sigmoid(2 - 2 * x))))
    jsv = js(p, q, dx)
    print("JS(p, q) =", jsv, " log 2 =", math.log(2))
    value = np.sum(p * np.log(dstar) + q * np.log(1 - dstar)) * dx
    print("V(D*) =", value, " 2 JS - log 4 =", 2 * jsv - math.log(4))
    half = np.full_like(x, 0.5)
    print("V(D = 1/2) =", np.sum(p * np.log(half) + q * np.log(1 - half)) * dx, " -log 4 =", -math.log(4))

    # Figure 2.
    for b in (-6.0, 0.0, 4.0):
        s = float(sigmoid(b))
        print(f"logit {b:+.0f}: D = {s:.6f}  |d sat| = {s:.6f}  |d non-sat| = {1 - s:.6f}")

    # Figure 3.
    xs = np.linspace(-6.0, 6.0, 4001)
    dxs = xs[1] - xs[0]
    pm = 0.5 * normal(xs, -3.0, 0.5) + 0.5 * normal(xs, 3.0, 0.5)
    qc = normal(xs, 3.0, 0.5)
    mean_p = 0.0
    var_p = 9.0 + 0.25
    wide = normal(xs, mean_p, math.sqrt(var_p))
    print("mixture variance:", var_p, " sd:", math.sqrt(var_p))
    print("JS(p, collapsed):", js(pm, qc, dxs), " JS(p, wide):", js(pm, wide, dxs))
    print("FID 1-D collapsed:", fid_1d(0.0, math.sqrt(var_p), 3.0, 0.5))
    print("FID 1-D wide gaussian:", fid_1d(0.0, math.sqrt(var_p), 0.0, math.sqrt(var_p)))
    mass_gap = float(np.sum(wide[np.abs(xs) < 1.5]) * dxs)
    print("wide gaussian mass in |x| < 1.5 (real mass there is ~0):", mass_gap)
    rng = np.random.default_rng(0)
    n = 600
    comp = rng.integers(0, 2, n)
    real = np.where(comp == 0, -3.0, 3.0) + 0.5 * rng.normal(size=n)
    collapsed = 3.0 + 0.5 * rng.normal(size=n)
    wide_s = math.sqrt(var_p) * rng.normal(size=n)
    good = np.where(rng.integers(0, 2, n) == 0, -3.0, 3.0) + 0.5 * rng.normal(size=n)
    for name, f in (("good", good), ("collapsed", collapsed), ("wide", wide_s)):
        pr, rc = precision_recall(real, f, k=3)
        print(f"{name}: precision {pr:.3f} recall {rc:.3f}")

    # Likelihood arithmetic from the lesson.
    print("log(0.01) =", math.log(0.01), " per dimension over 3072:", math.log(0.01) / 3072)

    # SVG point lists.
    def pl(px, py):
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))

    print("\nFIG1 (x 60..560 for -4..6, y 280..40 for 0..1.0 for D*, 0..0.45 for density)")
    idx = np.arange(0, len(x), 40)
    X1 = 60 + (x[idx] + 4) / 10 * 500
    print("p:", pl(X1, 280 - p[idx] / 0.45 * 240 * 0.55))
    print("q:", pl(X1, 280 - q[idx] / 0.45 * 240 * 0.55))
    print("D*:", pl(X1, 280 - dstar[idx] * 240))
    print("x of 0, 1, 2:", [60 + (v + 4) / 10 * 500 for v in (0, 1, 2)])

    print("\nFIG2 (x 60..560 for -8..8, y 280..40 for 0..1)")
    bb = np.linspace(-8, 8, 65)
    X2 = 60 + (bb + 8) / 16 * 500
    print("sat |grad| = D:", pl(X2, 280 - sigmoid(bb) * 240))
    print("non-sat |grad| = 1 - D:", pl(X2, 280 - (1 - sigmoid(bb)) * 240))
    print("x of -6:", 60 + 2 / 16 * 500)

    print("\nFIG3 (x 60..560 for -6..6, y 280..40)")
    idx = np.arange(0, len(xs), 80)
    X3 = 60 + (xs[idx] + 6) / 12 * 500
    top = qc.max()
    print("p:", pl(X3, 280 - pm[idx] / top * 220))
    print("collapsed:", pl(X3, 280 - qc[idx] / top * 220))
    print("wide:", pl(X3, 280 - wide[idx] / top * 220))
    print("peak p:", top, "peak wide:", wide.max())


if __name__ == "__main__":
    main()
