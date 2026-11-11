#!/usr/bin/env python3
"""Numbers for the modern-transformer-choices figures and worked examples.

Figure 1 is the map from query heads to key-value heads for three group
counts, with the key-value bytes per token per layer. Figure 2 is the depth
experiment: a stack of 32 residual MLP sublayers, wired as pre-norm and as
post-norm, with the stream's rms and the gradient reaching each layer's input.
Figure 3 is the parameter budget of GPT-2 small and Llama 3 8B.

The lesson also quotes the RMSNorm and LayerNorm rows for [1, 2, 3, 6], the
z-loss example, and the head and cache arithmetic. Re-run this file before
changing any of those numbers.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-5


def rms_forward(x, g):
    r = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + EPS)
    return x * r * g


def rms_backward(dy, x, g):
    r = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + EPS)
    xh = x * r
    u = dy * g
    return r * (u - xh * (u * xh).mean(-1, keepdims=True))


def stack_profile(kind, n_layers, d, seed, T=16):
    """Stream rms after each layer, and the input-gradient norm of each layer.

    Each sublayer is a ReLU MLP of width 4d. The gradient is measured against a
    random upstream gradient, and divided by that gradient's norm.
    """
    rng = np.random.default_rng(seed)
    layers = [
        (rng.normal(size=(d, 4 * d)) * np.sqrt(2 / d), rng.normal(size=(4 * d, d)) * np.sqrt(1 / (4 * d)))
        for _ in range(n_layers)
    ]
    g = np.ones(d)
    x = rng.normal(size=(T, d))
    dout = rng.normal(size=(T, d))
    caches, rms = [], []
    for W1, W2 in layers:
        if kind == "pre":
            a = rms_forward(x, g) @ W1
            caches.append((x, a))
            x = x + np.maximum(a, 0) @ W2
        else:
            a = x @ W1
            z = x + np.maximum(a, 0) @ W2
            caches.append((z, a))
            x = rms_forward(z, g)
        rms.append(float(np.sqrt((x**2).mean())))
    grads, dx = [], dout
    for (W1, W2), (u, a) in zip(reversed(layers), reversed(caches)):
        mask = a > 0
        if kind == "pre":
            dn = ((dx @ W2.T) * mask) @ W1.T
            dx = dx + rms_backward(dn, u, g)
        else:
            dz = rms_backward(dx, u, g)
            dx = dz + ((dz @ W2.T) * mask) @ W1.T
        grads.append(float(np.linalg.norm(dx) / np.linalg.norm(dout)))
    return np.array(rms), np.array(grads[::-1])


def block_params(d, n_heads, n_kv, d_ff, gated, bias, norm):
    dh = d // n_heads
    attn = d * n_heads * dh + 2 * d * n_kv * dh + n_heads * dh * d
    if bias:
        attn += n_heads * dh + 2 * n_kv * dh + d
    ffn = (3 if gated else 2) * d * d_ff
    if bias:
        ffn += (2 * d_ff + d) if gated else (d_ff + d)
    norms = 2 * d * (2 if norm == "layer" else 1)
    return attn, ffn, norms


def count_params(d, n_layers, n_heads, n_kv, d_ff, vocab, ctx, gated, bias, norm, tie):
    attn, ffn, norms = block_params(d, n_heads, n_kv, d_ff, gated, bias, norm)
    emb = vocab * d + ctx * d
    head = 0 if tie else vocab * d
    final = d * (2 if norm == "layer" else 1)
    return dict(emb=emb, attn=n_layers * attn, ffn=n_layers * ffn, norms=n_layers * norms + final, head=head)


def main():
    print("== RMSNorm and LayerNorm on [1, 2, 3, 6] ==")
    x = np.array([1.0, 2.0, 3.0, 6.0])
    print("mean square", (x**2).mean(), "rms", np.sqrt((x**2).mean() + EPS))
    print("rmsnorm", np.round(rms_forward(x, 1.0), 4))
    ln = (x - x.mean()) / np.sqrt(x.var() + EPS)
    print("layernorm", np.round(ln, 4))
    print("mean of rmsnorm row", round(float(rms_forward(x, 1.0).mean()), 4))
    rng = np.random.default_rng(0)
    for d in (64, 4096):
        v = rng.normal(size=d)
        print("d", d, "energy share of the mean direction, one draw", round(float(d * v.mean() ** 2 / (v**2).sum()), 5), "expected 1/d", round(1 / d, 5))

    print("\n== z-loss on logits [10, 12, 11] ==")
    z = np.array([10.0, 12.0, 11.0])
    logz = z.max() + np.log(np.exp(z - z.max()).sum())
    print("log Z", round(float(logz), 4), "log Z squared", round(float(logz**2), 3), "1e-4 * that", round(float(1e-4 * logz**2), 5))

    print("\n== QK-norm score bound ==")
    for dh in (64, 128):
        print("d_head", dh, "max |score| with unit-rms q and k:", round(float(np.sqrt(dh)), 2))

    print("\n== Figure 1: KV bytes per token per layer, fp16, d_head 128 ==")
    for G in (8, 2, 1):
        print("groups", G, "bytes", 2 * G * 128 * 2)
    print("Llama 2 7B (32 kv heads, 32 layers) bytes per token", 2 * 32 * 128 * 2 * 32)
    print("Llama 3 8B (8 kv heads, 32 layers) bytes per token", 2 * 8 * 128 * 2 * 32)
    print("4096 tokens, Llama 2 7B GiB", 2 * 32 * 128 * 2 * 32 * 4096 / 2**30)
    print("8192 tokens, Llama 3 8B GiB", 2 * 8 * 128 * 2 * 32 * 8192 / 2**30)
    print("heads 8, d_head 128: attention params with G = 8, 2, 1 (d = 1024):")
    for G in (8, 2, 1):
        print("  G", G, 1024 * 8 * 128 + 2 * 1024 * G * 128 + 8 * 128 * 1024)

    print("\n== Figure 2: depth profile, 32 layers, d = 64, mean of seeds 0 to 4 ==")
    for kind in ("pre", "post"):
        rms = np.mean([stack_profile(kind, 32, 64, s)[0] for s in range(5)], axis=0)
        grad = np.mean([stack_profile(kind, 32, 64, s)[1] for s in range(5)], axis=0)
        print(kind, "rms layers 1, 8, 16, 24, 32", np.round(rms[[0, 7, 15, 23, 31]], 3))
        print(kind, "grad layers 1, 8, 16, 24, 32", np.round(grad[[0, 7, 15, 23, 31]], 3))
        print(kind, "rms polyline", " ".join(f"{x},{y}" for x, y in zip(range(1, 33), np.round(rms, 3))))
        print(kind, "grad polyline", " ".join(f"{x},{y}" for x, y in zip(range(1, 33), np.round(grad, 3))))
    print("first-layer gradient against output gradient, mean of 5 seeds, and sqrt(L):")
    for L in (8, 16, 32, 64):
        pre = np.mean([stack_profile("pre", L, 64, s)[1][0] for s in range(5)])
        post = np.mean([stack_profile("post", L, 64, s)[1][0] for s in range(5)])
        print("  L", L, "pre", round(float(pre), 2), "post", round(float(post), 2), "sqrt(L)", round(float(np.sqrt(L)), 2))

    print("\n== Figure 3: parameter budgets ==")
    configs = {
        "GPT-2 small": dict(d=768, n_layers=12, n_heads=12, n_kv=12, d_ff=3072, vocab=50257, ctx=1024, gated=False, bias=True, norm="layer", tie=True),
        "Llama 2 7B": dict(d=4096, n_layers=32, n_heads=32, n_kv=32, d_ff=11008, vocab=32000, ctx=0, gated=True, bias=False, norm="rms", tie=False),
        "Llama 3 8B": dict(d=4096, n_layers=32, n_heads=32, n_kv=8, d_ff=14336, vocab=128256, ctx=0, gated=True, bias=False, norm="rms", tie=False),
    }
    for name, c in configs.items():
        parts = count_params(**c)
        total = sum(parts.values())
        shares = {k: round(100 * v / total, 1) for k, v in parts.items()}
        print(name, "total", total, parts, shares)
    c = dict(configs["Llama 3 8B"])
    c["n_kv"] = 32
    print("Llama 3 8B with full multi-head attention", sum(count_params(**c).values()))
    c = dict(configs["GPT-2 small"])
    c["tie"] = False
    print("GPT-2 small untied", sum(count_params(**c).values()))
    print("d_ff over d:", {n: round(c["d_ff"] / c["d"], 4) for n, c in configs.items()})
    print("8/3 d at d = 4096:", round(8 * 4096 / 3, 2), "rounded up to a multiple of 256:", int(np.ceil(8 * 4096 / 3 / 256) * 256))
    print("d / n_layers:", {n: round(c["d"] / c["n_layers"], 1) for n, c in configs.items()})


if __name__ == "__main__":
    main()
