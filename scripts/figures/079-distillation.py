#!/usr/bin/env python3
"""Numbers for the distillation figures.

Figure 1 draws the worked example: a teacher with logits [6, 3, 0] and a
student with logits [4, 3, 1], each softened at temperature 1 and 3.
Figure 2 draws the size of the gradient of the distillation loss with respect
to the student logits, with and without the T^2 factor, as T grows.

Prints the hand-check values, then the SVG coordinates the lesson quotes.
"""

from __future__ import annotations

import numpy as np

Z_T = np.array([6.0, 3.0, 0.0])
Z_S = np.array([4.0, 3.0, 1.0])


def softmax(z, T=1.0):
    a = np.asarray(z, float) / T
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def centred(z):
    return z - z.mean()


def experiment(seed, n=30, d=10, K=3, steps=400, lr=0.1, T=3.0):
    """Linear student, 30 examples. Same data as the exercise's last check."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, K))
    X = rng.normal(size=(n, d))
    X_test = rng.normal(size=(2000, d))
    p = softmax(X @ W)
    y = np.array([rng.choice(K, p=row) for row in p])
    z_t = X @ W + 0.3 * rng.normal(size=(n, K))
    truth = (X_test @ W).argmax(1)
    out = {}
    for name, alpha in (("hard", 0.0), ("soft", 1.0)):
        V = np.zeros((d, K))
        for _ in range(steps):
            g = alpha * T * (softmax(X @ V, T) - softmax(z_t, T)) + (1 - alpha) * (softmax(X @ V) - np.eye(K)[y])
            V = V - lr * X.T @ g / n
        out[name] = float(np.mean((X_test @ V).argmax(1) == truth))
    return out


def main():
    print("Worked example")
    for T in (1, 3):
        pt, ps = softmax(Z_T, T), softmax(Z_S, T)
        print(f"T={T}: teacher {pt.round(4)} student {ps.round(4)}")
        print(f"      KL(teacher||student)={kl(pt, ps):.4f}  T^2*KL={T * T * kl(pt, ps):.4f}")
        print(f"      ps-pt={(ps - pt).round(4)}")
    print(f"hard-label cross-entropy at T=1 for class 0: {-np.log(softmax(Z_S)[0]):.4f}")

    print("\nGradient norm against T (figure 2)")
    limit = np.linalg.norm((centred(Z_S) - centred(Z_T)) / 3)
    print(f"limit of the T^2 version: {limit:.4f}")
    # Plot area: x from 60 to 540 for T in 1..16 on a log axis, y from 190 (0) to 30 (0.8).
    for T in (1, 2, 4, 8, 16):
        g = softmax(Z_S, T) - softmax(Z_T, T)
        plain = np.linalg.norm(g / T)
        scaled = np.linalg.norm(T * g)
        x = 60 + 480 * np.log2(T) / 4
        print(f"T={T:2d} plain={plain:.4f} scaled={scaled:.4f}  x={x:.0f} "
              f"y_plain={190 - plain / 0.8 * 160:.1f} y_scaled={190 - scaled / 0.8 * 160:.1f}")

    print("\nGradients per logit with the T^2 factor")
    for T in (1, 3, 16):
        print(f"T={T}: {(T * (softmax(Z_S, T) - softmax(Z_T, T))).round(4)}")
    print(f"hard-label gradient at T=1: {(softmax(Z_S) - np.array([1.0, 0, 0])).round(4)}")
    print(f"limit (centred difference over K): {((centred(Z_S) - centred(Z_T)) / 3).round(4)}")

    print("\nExperiment: 30 examples, linear student, agreement with the true best class")
    runs = [experiment(s) for s in range(5)]
    for s, r in enumerate(runs):
        print(f"seed {s}: soft {r['soft']:.3f} hard {r['hard']:.3f}")
    print(f"mean soft {np.mean([r['soft'] for r in runs]):.3f} mean hard {np.mean([r['hard'] for r in runs]):.3f}")

    print("\nBar heights (figure 1), 120 units = probability 1")
    for T in (1, 3):
        print(T, (softmax(Z_T, T) * 120).round(1), (softmax(Z_S, T) * 120).round(1))


if __name__ == "__main__":
    main()
