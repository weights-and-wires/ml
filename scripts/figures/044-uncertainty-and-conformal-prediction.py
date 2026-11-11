#!/usr/bin/env python3
"""Numbers for the uncertainty and conformal prediction figures.

Figure 1: eight random-feature ridge models, each with its own seed, fitted to
the same 30 noisy points on x in [-2, 2]. The plot shows each member's curve,
the ensemble mean, and the ensemble spread, which widens outside the data.
Figure 2: coverage of a conformal interval over 200 seeded splits, with and
without the (n + 1) correction, at n = 10 calibration points and alpha = 0.1.

Prints every value the lesson quotes, then the SVG point lists it draws.
"""

from __future__ import annotations

import math

import numpy as np

WIDTH = 60
LAM = 1e-2


def make_data(rng: np.random.Generator, n: int = 30):
    x = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(2.0 * x) + 0.15 * rng.normal(size=n)
    return x, y


def fit_member(x: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=1.5, size=WIDTH)
    b = rng.uniform(-2.0, 2.0, size=WIDTH)

    def feats(t):
        return np.tanh(np.outer(np.asarray(t, dtype=float), w) + b)

    phi = feats(x)
    beta = np.linalg.solve(phi.T @ phi + LAM * np.eye(WIDTH), phi.T @ y)
    return lambda t: feats(t) @ beta


def quantile_corrected(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(np.sort(scores)[k - 1])


def quantile_naive(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = max(1, math.ceil(n * (1 - alpha)))
    return float(np.sort(scores)[k - 1])


def coverage_experiment(n_cal: int, alpha: float, quantile, splits: int = 200):
    rng = np.random.default_rng(7)
    covs = []
    for _ in range(splits):
        x_cal = rng.uniform(-2, 2, n_cal)
        y_cal = np.sin(x_cal) + 0.3 * rng.standard_t(5, n_cal)
        x_te = rng.uniform(-2, 2, 500)
        y_te = np.sin(x_te) + 0.3 * rng.standard_t(5, 500)
        q = quantile(np.abs(y_cal - np.sin(x_cal)), alpha)
        covs.append(np.mean(np.abs(y_te - np.sin(x_te)) <= q))
    return np.array(covs)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = make_data(rng)
    models = [fit_member(x, y, s) for s in range(8)]
    grid = np.linspace(-4.0, 4.0, 9)
    preds = np.stack([m(grid) for m in models])
    mean, sd = preds.mean(0), preds.std(0)
    print("grid     ", np.round(grid, 1))
    print("ens mean ", np.round(mean, 3))
    print("ens sd   ", np.round(sd, 3))
    inside = np.linspace(-2, 2, 81)
    outside = np.concatenate([np.linspace(-4, -2.5, 20), np.linspace(2.5, 4, 20)])
    pi = np.stack([m(inside) for m in models]).var(0).mean()
    po = np.stack([m(outside) for m in models]).var(0).mean()
    print(f"mean variance inside [-2, 2]: {pi:.4f}, outside [-4, -2.5] and [2.5, 4]: {po:.4f}")

    fine = np.linspace(-4.0, 4.0, 65)
    pf = np.stack([m(fine) for m in models])
    mf, sf = pf.mean(0), pf.std(0)

    def px(v):
        return 40 + (v + 4.0) / 8.0 * 480

    def py(v):
        return min(max(110 - v * 22, 6), 214)

    print("data points (svg):", " ".join(f"{px(a):.0f},{py(b):.0f}" for a, b in zip(x, y)))
    for i in (0, 3, 6):
        print(f"member {i} polyline:", " ".join(f"{px(a):.0f},{py(b):.0f}" for a, b in zip(fine, pf[i])))
    print("mean polyline:", " ".join(f"{px(a):.0f},{py(b):.0f}" for a, b in zip(fine, mf)))
    up = [f"{px(a):.0f},{py(b + 2 * s):.0f}" for a, b, s in zip(fine, mf, sf)]
    dn = [f"{px(a):.0f},{py(b - 2 * s):.0f}" for a, b, s in reversed(list(zip(fine, mf, sf)))]
    print("band polygon:", " ".join(up + dn))

    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.1, 1.5])
    print("hand example n=9 alpha=0.2 k =", math.ceil(10 * 0.8), "quantile", quantile_corrected(scores, 0.2))
    for alpha in (0.1,):
        good = coverage_experiment(10, alpha, quantile_corrected)
        bad = coverage_experiment(10, alpha, quantile_naive)
        print(f"n=10 alpha={alpha}: corrected mean coverage {good.mean():.3f} (theory {10 / 11:.3f}), naive {bad.mean():.3f} (theory {9 / 11:.3f})")
        for name, c in (("corrected", good), ("naive", bad)):
            hist, _ = np.histogram(c, bins=np.linspace(0.4, 1.0, 13))
            print(name, "histogram over bins of 0.05 from 0.40:", hist.tolist())
    big = coverage_experiment(200, 0.1, quantile_corrected)
    print(f"n=200 alpha=0.1 corrected mean coverage {big.mean():.3f}")
