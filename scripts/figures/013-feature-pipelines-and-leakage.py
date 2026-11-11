#!/usr/bin/env python3
"""Numbers for the feature-pipelines-and-leakage figures and worked examples.

Figure 1: cross-validated accuracy on pure noise (50 rows, 500 columns,
random 0/1 labels) when the 10 best columns are chosen on all rows (leaky) or
inside each fold (safe), averaged over 200 seeded datasets, for several
numbers of kept columns. The lesson SVG uses the printed "polyline" lines.

Also prints the numbers quoted in the prose: the duplicate-rows demo, the time
leakage demo, and the target-encoding correlation on noise.
"""

from __future__ import annotations

import numpy as np

N, D, SEEDS = 50, 500, 200


def select_top_k(X, y, k):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    corr = np.abs(Xc.T @ yc) / np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return np.argsort(-corr, kind="stable")[:k]


def centroid_predict(Xt, yt, Xs):
    c0, c1 = Xt[yt == 0].mean(axis=0), Xt[yt == 1].mean(axis=0)
    return (((Xs - c1) ** 2).sum(axis=1) < ((Xs - c0) ** 2).sum(axis=1)).astype(int)


def folds_for(n, k, seed):
    return [np.sort(p) for p in np.array_split(np.random.default_rng(seed).permutation(n), k)]


def cv(X, y, folds, keep_fn):
    accs = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        keep = keep_fn(X[train], y[train])
        accs.append(np.mean(centroid_predict(X[train][:, keep], y[train], X[test][:, keep]) == y[test]))
    return float(np.mean(accs))


def figure_one():
    ks = [1, 2, 5, 10, 20, 50, 100]
    leaky = np.zeros(len(ks))
    safe = np.zeros(len(ks))
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, D))
        y = rng.integers(0, 2, size=N)
        folds = folds_for(N, 5, 1000 + seed)
        for i, k in enumerate(ks):
            keep_all = select_top_k(X, y, k)
            leaky[i] += cv(X, y, folds, lambda Xt, yt: keep_all)
            safe[i] += cv(X, y, folds, lambda Xt, yt, k=k: select_top_k(Xt, yt, k))
    leaky /= SEEDS
    safe /= SEEDS
    print("n_keep:", ks)
    print("leaky mean accuracy:", np.round(leaky, 3).tolist())
    print("safe mean accuracy: ", np.round(safe, 3).tolist())
    # Plot box x 70..570 (index-spaced), y 240 (acc 0.4) .. 40 (acc 1.0).
    def px(i):
        return 70 + i * 500 / (len(ks) - 1)

    def py(a):
        return 240 - (a - 0.4) / 0.6 * 200

    for name, vals in (("leaky", leaky), ("safe", safe)):
        pts = " ".join(f"{px(i):.1f},{py(a):.1f}" for i, a in enumerate(vals))
        print(f"polyline {name}: {pts}")
    print("chance y:", round(py(0.5), 1))


def duplicates_demo():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(25, 20))
    X = np.vstack([base, base])           # every row appears twice
    y = rng.integers(0, 2, size=25)
    y = np.concatenate([y, y])            # twins share a label; the labels are pure noise
    random_acc, grouped_acc = [], []
    for seed in range(200):
        r = np.random.default_rng(seed)
        order = r.permutation(50)
        folds = [np.sort(p) for p in np.array_split(order, 5)]
        gorder = r.permutation(25)
        gfolds = [np.sort(np.concatenate([p, p + 25])) for p in np.array_split(gorder, 5)]
        for fl, store in ((folds, random_acc), (gfolds, grouped_acc)):
            correct = 0
            for test in fl:
                train = np.setdiff1d(np.arange(50), test)
                d = ((X[test][:, None, :] - X[train][None, :, :]) ** 2).sum(axis=2)
                correct += np.sum(y[train][d.argmin(axis=1)] == y[test])
            store.append(correct / 50)
    print(f"duplicates, random folds, 1-NN accuracy on noise: {np.mean(random_acc):.3f}")
    print(f"duplicates, grouped folds, 1-NN accuracy on noise: {np.mean(grouped_acc):.3f}")


def time_demo():
    t = np.arange(60, dtype=float)
    rng = np.random.default_rng(0)
    y = 0.5 * t + np.cumsum(rng.normal(size=60))
    # Random folds: predict a day by the mean of the day before and the day after (both often in training).
    mse_rand = []
    for seed in range(200):
        r = np.random.default_rng(seed)
        test = np.sort(r.choice(np.arange(1, 59), size=12, replace=False))
        pred = (y[test - 1] + y[test + 1]) / 2
        mse_rand.append(np.mean((pred - y[test]) ** 2))
    # Forward split: train days 0..47, test days 48..59, predict each by the last training value.
    pred = np.full(12, y[47])
    mse_fwd = np.mean((pred - y[48:]) ** 2)
    print(f"time leakage, neighbour interpolation, random test days: MSE {np.mean(mse_rand):.2f}")
    print(f"time leakage, last-known-value forecast, forward split: MSE {mse_fwd:.2f}")


def target_encoding_demo():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 150, size=300)
    y = rng.integers(0, 2, size=300).astype(float)
    prior = y.mean()
    table = {}
    for c in np.unique(ids):
        rows = ids == c
        table[c] = (y[rows].sum() + prior) / (rows.sum() + 1)
    naive = np.array([table[c] for c in ids])
    folds = [np.arange(i, 300, 5) for i in range(5)]
    oof = np.empty(300)
    for test in folds:
        train = np.setdiff1d(np.arange(300), test)
        p = y[train].mean()
        tab = {}
        for c in np.unique(ids[train]):
            rows = ids[train] == c
            tab[c] = (y[train][rows].sum() + p) / (rows.sum() + 1)
        oof[test] = [tab.get(c, p) for c in ids[test]]
    print(f"target encoding on noise: all-rows corr {np.corrcoef(naive, y)[0, 1]:.3f}, "
          f"out-of-fold corr {np.corrcoef(oof, y)[0, 1]:.3f}")


if __name__ == "__main__":
    figure_one()
    duplicates_demo()
    time_demo()
    target_encoding_demo()
