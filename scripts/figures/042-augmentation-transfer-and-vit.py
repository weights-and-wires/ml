"""Numbers drawn in the figures of 041-augmentation-transfer-and-vit.

Stdlib and numpy only. Prints every value the lesson's figures and worked
examples use.
"""

import numpy as np

# Figure 1: patchify a 6 by 6 one-channel image with p = 2.
x = np.arange(36, dtype=float).reshape(1, 6, 6)
tokens = x.reshape(1, 3, 2, 3, 2).transpose(1, 3, 0, 2, 4).reshape(9, 4)
print("patch tokens of arange(36) on 6x6, p=2:")
for i, t in enumerate(tokens):
    print(f"  token {i}: {t.astype(int).tolist()}")
print("ViT patches at 224x224, p=16:", (224 // 16) ** 2, "tokens, each", 16 * 16 * 3, "numbers")
print("ViT-Base patch embedding weights:", 16 * 16 * 3 * 768, "plus", 768, "biases")

# Figure 2: weights of a 3x3 full layer versus a depthwise-separable layer.
print("channels c, full, separable, ratio, 1/c + 1/9:")
for c in (16, 32, 64, 128):
    full = c * c * 9
    sep = c * 9 + c * c
    print(f"  {c}: {full}, {sep}, {sep / full:.4f}, {1 / c + 1 / 9:.4f}")

# Figure 3 and the worked mixup example.
xa, xb = np.array([1.0, 0.0]), np.array([0.0, 2.0])
ya, yb = np.array([1.0, 0.0]), np.array([0.0, 1.0])
for lam in (1.0, 0.75, 0.5, 0.25, 0.0):
    print(f"lam {lam}: x = {(lam * xa + (1 - lam) * xb).tolist()}, y = {(lam * ya + (1 - lam) * yb).tolist()}")

# Transfer example from the exercise's data: frozen random features plus a ridge head.
def make_bars(n, rng, size=8, noise=0.3):
    X = np.zeros((n, size, size)); y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % 2
        c = rng.integers(1, size - 1)
        if y[i] == 0: X[i, :, c] = 1.0
        else: X[i, c, :] = 1.0
    return X + noise * rng.normal(size=X.shape), y

def fit(F, y, lam):
    F1 = np.hstack([F, np.ones((len(F), 1))]); t = 2.0 * y - 1
    n, d = F1.shape
    return np.linalg.solve(F1.T @ F1 / n + lam * np.eye(d), F1.T @ t / n)

for width in (16, 64, 256):
    accs, bases = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        Xtr, ytr = make_bars(200, rng); Xte, yte = make_bars(200, rng)
        W = np.random.default_rng(100 + seed).normal(size=(64, width)) / 8.0
        feats = lambda X: np.maximum(X.reshape(len(X), -1) @ W, 0.0)
        w = fit(feats(Xtr), ytr, 1e-2)
        F1 = np.hstack([feats(Xte), np.ones((len(Xte), 1))])
        accs.append(np.mean((F1 @ w > 0) == yte))
        m = Xtr.mean(axis=(1, 2)); th = np.median(m)
        bases.append(max(np.mean((m > th) == ytr), np.mean((m <= th) == ytr)))
    print(f"width {width}: held-out accuracy {np.mean(accs):.3f}, threshold {np.mean(bases):.3f}")
