#!/usr/bin/env python3
"""Numbers and SVG coordinates for the graph neural network lesson.

Figure 1 draws the four-node star graph of the worked example with the
symmetric normalised adjacency on its edges.
Figure 2 draws six scalar node features on a path as the layers stack.

The script also prints the worked arithmetic, the aggregation examples, the
permutation check, the over-smoothing curve, and the two-community run quoted
in the lesson. Uses only numpy.
"""

from __future__ import annotations

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def norm_adj(A):
    At = A + np.eye(len(A))
    d = At.sum(1)
    return At / np.sqrt(d[:, None] * d[None, :])


def spread(x):
    return float(np.mean(x.max(0) - x.min(0)))


def main():
    # Worked example: star with centre node 1 and leaves 0, 2, 3.
    A = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], float)
    deg = (A + np.eye(4)).sum(1)
    Ah = norm_adj(A)
    print("A+I degrees", deg)
    print("A_hat\n", Ah)
    print("row-normalised D^-1(A+I)\n", (A + np.eye(4)) / deg[:, None])
    print("A_hat @ sqrt(d) == sqrt(d):", Ah @ np.sqrt(deg), np.sqrt(deg))
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    W = np.array([[1.0], [-1.0]])
    pre = Ah @ H @ W
    print("A_hat H\n", Ah @ H)
    print("pre-activation", pre.ravel(), "relu", np.maximum(pre, 0).ravel())

    # Aggregation examples (scalar neighbour features).
    for name, f in (("sum", np.sum), ("mean", np.mean), ("max", np.max)):
        print(name, "{1,3}", f([1, 3]), "{2,2}", f([2, 2]), "{1,1}", f([1, 1]), "{1}", f([1]),
              "{1,2}", f([1, 2]), "{1,1,2,2}", f([1, 1, 2, 2]), "{2,3}", f([2, 3]))
    # Message passing on the star with scalar features.
    h = np.array([1.0, 2.0, 3.0, 4.0])
    print("star sum", A @ h, "mean", (A @ h) / A.sum(1), "max", [h[A[i] > 0].max() for i in range(4)])

    # Permutation check.
    rng = np.random.default_rng(0)
    perm = rng.permutation(4)
    P = np.eye(4)[perm]
    Wr = rng.normal(size=(2, 3))
    out = np.maximum(Ah @ H @ Wr, 0)
    out_p = np.maximum(norm_adj(P @ A @ P.T) @ (P @ H) @ Wr, 0)
    print("perm", perm, "equivariant error", np.abs(out_p - P @ out).max(),
          "readout error", np.abs(out_p.mean(0) - out.mean(0)).max())
    flat = A.ravel()
    print("flattened adjacency changes:", flat, (P @ A @ P.T).ravel())

    # WL: a 6-cycle and two triangles.
    C6 = np.zeros((6, 6))
    for i in range(6):
        C6[i, (i + 1) % 6] = C6[(i + 1) % 6, i] = 1
    T2 = np.zeros((6, 6))
    for a, b in ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)):
        T2[a, b] = T2[b, a] = 1
    x = np.ones((6, 1))
    for _ in range(3):
        print("c6", (C6 @ x).ravel(), "two triangles", (T2 @ x).ravel()); x = C6 @ x / 2 + 0 * x
    # Over-smoothing on a path of six nodes.
    n = 6
    Pth = np.zeros((n, n))
    for i in range(n - 1):
        Pth[i, i + 1] = Pth[i + 1, i] = 1
    Ap = norm_adj(Pth)
    d = (Pth + np.eye(n)).sum(1)
    x0 = np.array([[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]]) * 1.0 + np.array([[-1.0], [2.0], [0.5], [-0.5], [1.5], [-2.0]]) * 0
    x0 = np.array([[3.0], [-1.0], [2.0], [-2.0], [1.0], [-3.0]])
    xs = [x0]
    for _ in range(12):
        xs.append(Ap @ xs[-1])
    print("path degrees", d)
    print("spread by layer (rescaled by sqrt(d))",
          [round(spread(x / np.sqrt(d)[:, None]), 4) for x in xs])
    print("raw spread by layer", [round(spread(x), 4) for x in xs])
    print("rescaled trajectories (node x layer):")
    for i in range(n):
        print(i, [round(float(x[i, 0] / np.sqrt(d[i])), 3) for x in xs])
    print("limit", np.round(float(((x0[:, 0] * np.sqrt(d)).sum() / d.sum())), 4))
    S = [spread(np.linalg.matrix_power(Ap, k) @ x0 / np.sqrt(d)[:, None]) for k in range(0, 41, 5)]
    print("path spread at 0,5,...,40", np.round(S, 4))

    # Two-community graph.
    rng = np.random.default_rng(3)
    nn, half = 60, 30
    y = np.repeat([0, 1], half)
    prob = np.where(y[:, None] == y[None, :], 0.25, 0.02)
    U = np.triu(rng.uniform(size=(nn, nn)) < prob, 1)
    As = (U | U.T).astype(float)
    X = np.stack([y * 0.6 - 0.3, -(y * 0.6 - 0.3)], 1) + rng.normal(size=(nn, 2)) * 1.0
    print("sbm edges", int(As.sum() / 2), "within", int((As * (y[:, None] == y[None, :])).sum() / 2))
    # feature-only nearest class mean from 3 labels per class
    lab = np.array([0, 1, 2, 30, 31, 32])
    mask = np.ones(nn, bool); mask[lab] = False
    m0, m1 = X[lab[:3]].mean(0), X[lab[3:]].mean(0)
    pred = (np.linalg.norm(X - m1, axis=1) < np.linalg.norm(X - m0, axis=1)).astype(int)
    print("feature-only accuracy on unlabelled", (pred[mask] == y[mask]).mean())


if __name__ == "__main__":
    main()
