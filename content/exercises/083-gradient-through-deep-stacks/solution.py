"""Reference solution: how much gradient survives a deep stack."""

import numpy as np


def orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    # QR alone is biased; fixing the signs of the columns makes q evenly random.
    return q * np.sign(np.diag(r))


def scalar_gradient_factor(J: float, depth: int, residual: bool) -> float:
    factor = 1.0 + J if residual else J
    return float(factor**depth)


def gradient_norm_through_stack(depth: int, residual: bool) -> float:
    d = 16
    scale = 0.1
    rng = np.random.default_rng(0)

    layers = [scale * orthogonal_matrix(d, rng) for _ in range(depth)]

    g = np.ones(d) / np.sqrt(d)
    for w in reversed(layers):
        g = g + g @ w if residual else g @ w
    return float(np.linalg.norm(g))


if __name__ == "__main__":
    print("depth   residual length   plain length   scalar (1+J)^L   scalar J^L")
    for depth in (1, 2, 10, 20, 50):
        r = gradient_norm_through_stack(depth, residual=True)
        p = gradient_norm_through_stack(depth, residual=False)
        sr = scalar_gradient_factor(0.1, depth, residual=True)
        sp = scalar_gradient_factor(0.1, depth, residual=False)
        print(f"{depth:5d}   {r:15.4f}   {p:12.3e}   {sr:14.4g}   {sp:10.3e}")

    rng = np.random.default_rng(1)
    q = orthogonal_matrix(16, rng)
    v = rng.normal(size=16)
    print(f"\nlength of v: {np.linalg.norm(v):.6f}, length of v @ Q: {np.linalg.norm(v @ q):.6f}")
