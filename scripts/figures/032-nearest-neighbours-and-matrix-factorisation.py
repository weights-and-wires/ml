#!/usr/bin/env python3
"""Numbers for the nearest-neighbours and matrix-factorisation figures.

Prints every value the lesson quotes: the five-point k-NN example, the leave-one-out
error for several k, the distance ratio against dimension, the overlap between
users on a sparse table, and held-out RMSE for the baselines, user-based k-NN,
and alternating least squares (ALS) on a planted rank-2 table.

Run it with `--svg DIR` to also write the figure bodies the lesson uses.
"""

from __future__ import annotations

import sys

import numpy as np


def sq_dists(A, B):
    d = (A**2).sum(1)[:, None] - 2 * A @ B.T + (B**2).sum(1)[None, :]
    return np.maximum(d, 0.0)


def knn_idx(Xtr, Xte, k):
    return np.argsort(sq_dists(Xte, Xtr), axis=1, kind="stable")[:, :k]


def loo_error(X, y, k):
    d = sq_dists(X, X)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    votes = np.array([np.bincount(y[r], minlength=2) for r in idx])
    return float(np.mean(votes.argmax(1) != y))


def distance_ratio(d, n, rng):
    X = rng.uniform(size=(n, d))
    q = rng.uniform(size=(1, d))
    dist = np.sqrt(sq_dists(q, X))[0]
    return float(dist.min() / dist.max())


def masked_loss(R, M, U, V, lam):
    r = (R - U @ V.T) * M
    return float((r**2).sum() + lam * ((U**2).sum() + (V**2).sum()))


def als_step(R, M, U, V, lam):
    out = np.empty_like(U)
    k = U.shape[1]
    for i in range(R.shape[0]):
        o = M[i] > 0
        Vo = V[o]
        out[i] = np.linalg.solve(Vo.T @ Vo + lam * np.eye(k), Vo.T @ R[i, o])
    return out


def fit_als(R, M, k, lam, iters, rng):
    U = 0.1 * rng.normal(size=(R.shape[0], k))
    V = 0.1 * rng.normal(size=(R.shape[1], k))
    hist = [masked_loss(R, M, U, V, lam)]
    for _ in range(iters):
        U = als_step(R, M, U, V, lam)
        V = als_step(R.T, M.T, V, U, lam)
        hist.append(masked_loss(R, M, U, V, lam))
    return U, V, hist


def rmse(R, M, P):
    return float(np.sqrt(((R - P) ** 2)[M > 0].mean()))


def user_knn_predict(R, M, test_M, k):
    """User-based k-NN: mean squared gap over co-rated items, k nearest raters."""
    n, m = R.shape
    mu = (R * M).sum() / M.sum()
    co = M @ M.T
    sq = ((R * M) ** 2) @ M.T
    cross = (R * M) @ (R * M).T
    gap = sq + sq.T - 2 * cross  # sum over co-rated items of (r_ui - r_vi)^2
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(co > 0, gap / co, np.inf)
    np.fill_diagonal(dist, np.inf)
    P = np.full((n, m), mu)
    for u in range(n):
        order = np.argsort(dist[u], kind="stable")
        for j in np.where(test_M[u] > 0)[0]:
            nb = [v for v in order if M[v, j] > 0 and np.isfinite(dist[u, v])][:k]
            if nb:
                P[u, j] = R[nb, j].mean()
    return P


def main():
    out = {}
    # 1. five-point worked example
    Xtr = np.array([[0, 0], [1, 0], [0, 2], [3, 3], [4, 3]], float)
    ytr = np.array([0, 0, 1, 1, 1])
    price = np.array([10.0, 12.0, 20.0, 30.0, 34.0])
    q = np.array([[1.0, 1.0]])
    d = sq_dists(q, Xtr)[0]
    print("example squared distances:", d)
    for k in (1, 2, 3):
        idx = knn_idx(Xtr, q, k)[0]
        print(f"k={k} neighbours {idx} labels {ytr[idx]} mean price {price[idx].mean()}")
    out["ex"] = (Xtr, ytr, q[0])

    # 2. choose k on overlapping blobs
    rng = np.random.default_rng(0)
    n = 200
    y = np.repeat([0, 1], n // 2)
    X = np.concatenate([rng.normal([0, 0], 1.2, (n // 2, 2)), rng.normal([2, 1.5], 1.2, (n // 2, 2))])
    ks = [1, 3, 5, 9, 15, 25, 51, 101]
    errs = [loo_error(X, y, k) for k in ks]
    print("LOO error by k:", dict(zip(ks, np.round(errs, 3))))
    # the same with the self-match left in
    dd = sq_dists(X, X)
    idx = np.argsort(dd, axis=1, kind="stable")[:, :1]
    print("k=1 error when the point matches itself:", float(np.mean(y[idx[:, 0]] != y)))

    # 3. distance concentration
    dims = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ratios = []
    for dim in dims:
        r = np.mean([distance_ratio(dim, 200, np.random.default_rng(s)) for s in range(20)])
        ratios.append(float(r))
    print("mean nearest/farthest ratio by d:", dict(zip(dims, np.round(ratios, 3))))
    out["ratio"] = (dims, ratios)

    # 4. planted rank-2 rating table
    rng = np.random.default_rng(1)
    nu, ni, rank = 120, 80, 2
    Ut = rng.normal(size=(nu, rank))
    Vt = rng.normal(size=(ni, rank))
    R = Ut @ Vt.T + 0.2 * rng.normal(size=(nu, ni))
    dens = 0.15
    obs = rng.uniform(size=(nu, ni)) < dens
    split = rng.uniform(size=(nu, ni)) < 0.8
    Mtr = (obs & split).astype(float)
    Mte = (obs & ~split).astype(float)
    print("observed fraction:", obs.mean(), "train entries:", int(Mtr.sum()), "test entries:", int(Mte.sum()))
    co = Mtr @ Mtr.T
    iu = np.triu_indices(nu, 1)
    print("mean co-rated items per user pair:", co[iu].mean(), "pairs with fewer than 3:", float((co[iu] < 3).mean()))
    mu = (R * Mtr).sum() / Mtr.sum()
    base = np.full((nu, ni), mu)
    cnt = Mtr.sum(1)
    bu = np.where(cnt > 0, ((R - mu) * Mtr).sum(1) / np.maximum(cnt, 1), 0.0)
    ubias = mu + bu[:, None]
    P_knn = user_knn_predict(R, Mtr, Mte, 10)
    print("RMSE global mean:", rmse(R, Mte, base))
    print("RMSE user bias:  ", rmse(R, Mte, ubias))
    print("RMSE user kNN k=10:", rmse(R, Mte, P_knn))
    res = {}
    for k_, lam in ((2, 1.0), (2, 5.0), (2, 0.1), (5, 1.0), (1, 1.0)):
        U, V, h = fit_als(R, Mtr, k_, lam, 25, np.random.default_rng(0))
        res[(k_, lam)] = (rmse(R, Mte, U @ V.T), rmse(R, Mtr, U @ V.T))
        print(f"ALS rank {k_} lam {lam}: test {res[(k_, lam)][0]:.4f} train {res[(k_, lam)][1]:.4f}")
    U, V, h = fit_als(R, Mtr, 2, 1.0, 25, np.random.default_rng(0))
    print("loss history first 6:", np.round(h[:6], 1), "last", round(h[-1], 1), "monotone", all(a >= b - 1e-9 for a, b in zip(h, h[1:])))
    # test RMSE after each sweep
    sweep = []
    U = 0.1 * np.random.default_rng(0).normal(size=(nu, 2))
    V = 0.1 * np.random.default_rng(0).normal(size=(ni, 2))
    rng0 = np.random.default_rng(0)
    U = 0.1 * rng0.normal(size=(nu, 2)); V = 0.1 * rng0.normal(size=(ni, 2))
    for s in range(30):
        U = als_step(R, Mtr, U, V, 1.0); V = als_step(R.T, Mtr.T, V, U, 1.0)
        sweep.append(rmse(R, Mte, U @ V.T))
    print("ALS held-out RMSE after sweeps 1..30:", np.round(sweep, 3))
    out["rmse"] = (rmse(R, Mte, base), rmse(R, Mte, ubias), rmse(R, Mte, P_knn), sweep)

    # 5. hand example for one ALS row
    V1 = np.array([1.0, 2.0, 3.0])
    r = np.array([2.0, np.nan, 5.0])
    o = ~np.isnan(r)
    u = (V1[o] @ r[o]) / (V1[o] @ V1[o] + 1.0)
    print("hand ALS row: u =", u, "pred item 1:", u * 2.0)
    u0 = (V1[o] @ r[o]) / (V1[o] @ V1[o])
    print("without ridge u =", u0)
    # fit zeros wrongly
    r0 = np.array([2.0, 0.0, 5.0])
    print("fitting the zero too: u =", (V1 @ r0) / (V1 @ V1 + 1.0))

    if len(sys.argv) == 3 and sys.argv[1] == "--svg":
        write_svgs(sys.argv[2], out)


def write_svgs(path, out):
    import os
    os.makedirs(path, exist_ok=True)
    # Figure 1: five-point example
    Xtr, ytr, q = out["ex"]
    sx = lambda x: 70 + x * 75
    sy = lambda y: 210 - y * 55
    s = ['<svg viewBox="0 0 560 250" role="img" aria-label="Five labelled training points and a query at (1, 1) with a circle through its three nearest neighbours">',
         '<rect x="40" y="20" width="480" height="210" class="fill-sunk stroke-grid"/>']
    r = np.sqrt(2.0) * 75
    s.append(f'<circle cx="{sx(q[0])}" cy="{sy(q[1])}" r="{r:.1f}" class="fill-soft-b stroke-b dashed"/>')
    for i, (p, lab) in enumerate(zip(Xtr, ytr)):
        cls = "fill-a" if lab == 0 else "fill-c"
        s.append(f'<circle cx="{sx(p[0])}" cy="{sy(p[1])}" r="7" class="{cls}"/>')
        s.append(f'<text x="{sx(p[0]) + 11}" y="{sy(p[1]) - 9}" class="fig-small">#{i}</text>')
    s.append(f'<rect x="{sx(q[0]) - 7}" y="{sy(q[1]) - 7}" width="14" height="14" class="fill-b"/>')
    s.append(f'<text x="{sx(q[0]) + 11}" y="{sy(q[1]) + 22}" class="fig-bold">query</text>')
    s.append('<text x="340" y="192" class="fig-small">blue: class 0, green: class 1</text>')
    s.append('<text x="340" y="210" class="fig-small">circle: k = 3 reach</text></svg>')
    open(os.path.join(path, "fig1.svg"), "w").write("\n".join(s))
    # Figure 2: ratio vs d
    dims, ratios = out["ratio"]
    x0, x1, y0, y1 = 70, 520, 200, 30
    lx = lambda d: x0 + (np.log10(d) / 3.0) * (x1 - x0)
    ly = lambda r: y0 - r * (y0 - y1)
    s = ['<svg viewBox="0 0 560 250" role="img" aria-label="Mean ratio of nearest to farthest distance against dimension, rising from near 0 toward 1">',
         f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>',
         f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="stroke-muted"/>']
    for t in (0, 0.5, 1.0):
        s.append(f'<line x1="{x0}" y1="{ly(t):.1f}" x2="{x1}" y2="{ly(t):.1f}" class="stroke-grid"/>')
        s.append(f'<text x="{x0 - 8}" y="{ly(t) + 4:.1f}" text-anchor="end" class="fig-small">{t:g}</text>')
    for d in (1, 10, 100, 1000):
        s.append(f'<text x="{lx(d):.1f}" y="{y0 + 18}" text-anchor="middle" class="fig-small">{d}</text>')
    pts = " ".join(f"{lx(d):.1f},{ly(r):.1f}" for d, r in zip(dims, ratios))
    s.append(f'<polyline points="{pts}" fill="none" class="stroke-b"/>')
    for d, r in zip(dims, ratios):
        s.append(f'<circle cx="{lx(d):.1f}" cy="{ly(r):.1f}" r="4" class="fill-b"/>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 40}" text-anchor="middle" class="fig-small">dimension d (log scale)</text>')
    s.append(f'<text x="{x0 + 8}" y="{y1 + 4}" class="fig-small">nearest / farthest distance</text></svg>')
    open(os.path.join(path, "fig2.svg"), "w").write("\n".join(s))
    # Figure 3: held-out RMSE
    g, ub, kn, sweep = out["rmse"]
    top = max(g, ub, kn) * 1.1
    bx0, bx1, by0, by1 = 70, 520, 200, 30
    ly = lambda v: by0 - v / top * (by0 - by1)
    lx = lambda i: bx0 + (i / 29.0) * (bx1 - bx0 - 20) + 10
    s = ['<svg viewBox="0 0 560 250" role="img" aria-label="Held-out RMSE of alternating least squares after each sweep, against three flat baselines">',
         f'<line x1="{bx0}" y1="{by0}" x2="{bx1}" y2="{by0}" class="stroke-muted"/>',
         f'<line x1="{bx0}" y1="{by0}" x2="{bx0}" y2="{by1}" class="stroke-muted"/>']
    legend = (("global mean", g, "stroke-muted"), ("user bias", ub, "stroke-d"), ("user k-NN, k = 10", kn, "stroke-c"))
    for n_, (name, v, cls) in enumerate(legend):
        s.append(f'<line x1="{bx0}" y1="{ly(v):.1f}" x2="{bx1}" y2="{ly(v):.1f}" class="{cls} dashed"/>')
        s.append(f'<line x1="{(70, 205, 320)[n_]}" y1="236" x2="{(70, 205, 320)[n_] + 20}" y2="236" class="{cls} dashed"/>')
        s.append(f'<text x="{(70, 205, 320)[n_] + 25}" y="240" class="fig-small">{name} {v:.2f}</text>')
    pts = " ".join(f"{lx(i):.1f},{ly(v):.1f}" for i, v in enumerate(sweep))
    s.append(f'<polyline points="{pts}" fill="none" class="stroke-b"/>')
    for i, v in enumerate(sweep):
        s.append(f'<circle cx="{lx(i):.1f}" cy="{ly(v):.1f}" r="3.5" class="fill-b"/>')
    s.append(f'<text x="{lx(29) - 4:.1f}" y="{ly(sweep[-1]) - 10:.1f}" text-anchor="end" class="fig-bold">ALS {sweep[-1]:.2f}</text>')
    for t in (0, 1, 2):
        v = t * 0.5 if top > 1.5 else t * 0.5
    for v in (0.0, 0.5, 1.0, 1.5):
        if v <= top:
            s.append(f'<text x="{bx0 - 8}" y="{ly(v) + 4:.1f}" text-anchor="end" class="fig-small">{v:g}</text>')
    s.append(f'<text x="{bx0 + 40}" y="{by0 + 18}" class="fig-small">ALS sweeps 1 to 30</text>')
    s.append(f'<text x="{bx0 + 8}" y="{by1 - 6}" class="fig-small">held-out RMSE</text></svg>')
    open(os.path.join(path, "fig3.svg"), "w").write("\n".join(s))


if __name__ == "__main__":
    main()
