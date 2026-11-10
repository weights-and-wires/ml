#!/usr/bin/env python3
"""Numbers for the contrastive-learning figures.

Figure 1 draws the worked example: three unit-length embeddings per view, their
similarity matrix, and the row softmax at temperature 0.5.
Figure 2 draws the 2-D embeddings of a linear encoder trained on planted
clusters, once with InfoNCE and once with an alignment-only loss, on the unit
circle. The alignment-only encoder falls onto a line.

Prints the hand-check values, then the SVG coordinates the lesson quotes.
"""

from __future__ import annotations

import numpy as np

C, D, N = 4, 12, 64
MEANS = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]], float)


def norm(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def lse(a):
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def info_nce(z1, z2, tau):
    logits = norm(z1) @ norm(z2).T / tau
    return float(np.mean(lse(logits) - np.diag(logits)))


def back(z, n, d):
    return (d - n * np.sum(n * d, 1, keepdims=True)) / np.linalg.norm(z, axis=1, keepdims=True)


def info_nce_grad(z1, z2, tau):
    n1, n2 = norm(z1), norm(z2)
    logits = n1 @ n2.T / tau
    p = np.exp(logits - lse(logits)[:, None])
    g = (p - np.eye(len(z1))) / (len(z1) * tau)
    return back(z1, n1, g @ n2), back(z2, n2, g.T @ n1)


def data(rng, n_per):
    y = np.repeat(np.arange(C), n_per)
    x = np.zeros((len(y), D))
    x[:, :2] = MEANS[y] + 0.5 * rng.normal(size=(len(y), 2))
    x[:, 2:] = 3 * rng.normal(size=(len(y), D - 2))
    return x, y


def views(x, rng):
    out = []
    for _ in range(2):
        v = x.copy()
        v[:, :2] += 0.3 * rng.normal(size=(len(x), 2))
        v[:, 2:] = 3 * rng.normal(size=(len(x), D - 2))
        out.append(v)
    return out


def train(xtr, tau, steps, lr, seed, negatives):
    rng = np.random.default_rng(seed)
    w = 0.1 * rng.normal(size=(D, 2))
    for _ in range(steps):
        v1, v2 = views(xtr[rng.choice(len(xtr), N, replace=False)], rng)
        z1, z2 = v1 @ w, v2 @ w
        if negatives:
            d1, d2 = info_nce_grad(z1, z2, tau)
        else:
            n1, n2 = norm(z1), norm(z2)
            g = 2 * (n1 - n2) / N
            d1, d2 = back(z1, n1, g), back(z2, n2, -g)
        w -= lr * (v1.T @ d1 + v2.T @ d2)
    return w


def knn(ztr, ytr, zte, yte, k=5):
    s = norm(zte) @ norm(ztr).T
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    pred = np.array([np.bincount(ytr[r], minlength=C).argmax() for r in idx])
    return float((pred == yte).mean())


def uniformity(z, t=2.0):
    n = norm(z)
    d2 = ((n[:, None] - n[None]) ** 2).sum(-1)
    iu = np.triu_indices(len(n), 1)
    return float(np.log(np.mean(np.exp(-t * d2[iu]))))


def main() -> None:
    z1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    z2 = np.array([[0.8, 0.6], [0.6, 0.8], [0.0, 1.0]])
    tau = 0.5
    sim = z1 @ z2.T
    logits = sim / tau
    p = np.exp(logits - lse(logits)[:, None])
    print("similarity\n", sim)
    print("logits\n", logits)
    print("row softmax\n", np.round(p, 4))
    print("row losses", np.round(-np.log(np.diag(p)), 4))
    print("info_nce(z1, z2) =", round(info_nce(z1, z2, tau), 4))
    print("info_nce(z2, z1) =", round(info_nce(z2, z1, tau), 4))
    print("clip_loss        =", round(0.5 * (info_nce(z1, z2, tau) + info_nce(z2, z1, tau)), 4))
    print("log 3            =", round(float(np.log(3)), 4))
    big = np.array([[2.0, 0.0], [0.0, 3.0], [0.3, 0.4]])
    print("info_nce with unnormalised rows (same directions) =", round(info_nce(big, z2, tau), 4))

    rng = np.random.default_rng(0)
    xtr, ytr = data(rng, 50)
    xte, yte = data(rng, 25)
    results = {}
    for name, negs, steps in [("infonce", True, 100), ("alignment-only", False, 5000)]:
        w = train(xtr, tau, steps, 0.2, 1, negs)
        ztr, zte = xtr @ w, xte @ w
        results[name] = zte
        v1, v2 = views(xte, np.random.default_rng(7))
        align = float(np.mean(np.sum((norm(v1 @ w) - norm(v2 @ w)) ** 2, axis=1)))
        print(
            f"{name}: steps={steps} knn={knn(ztr, ytr, zte, yte):.3f} "
            f"uniformity={uniformity(ztr):.3f} alignment={align:.4f}"
        )
    print("constant encoder knn =", knn(np.ones((200, 2)), ytr, np.ones((100, 2)), yte))

    cx, cy, r = {"infonce": 150, "alignment-only": 450}, 150, 100
    for name, zte in results.items():
        n = norm(zte)
        print(f"-- circle points: {name}")
        for c in range(C):
            pts = n[yte == c][:12]
            print(c, " ".join(f"{cx[name] + r * x:.0f},{cy - r * y:.0f}" for x, y in pts))
        angles = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
        print("angle range per class:", [(round(angles[yte == c].min()), round(angles[yte == c].max())) for c in range(C)])


if __name__ == "__main__":
    main()
