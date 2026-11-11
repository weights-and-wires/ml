"""Numbers behind the figures in 076-evaluating-language-models.

Prints the values that the lesson's two figures and worked examples draw.
Uses only the standard library and numpy.
"""

import math

import numpy as np

LN2 = math.log(2.0)

# Figure 1: one 40-byte text, two tokenisers, the same total loss.
n_bytes = 40
total_nats = n_bytes * LN2  # exactly 1 bit per byte
for name, n_tokens in (("tokeniser A", 10), ("tokeniser B", 20)):
    per_token = total_nats / n_tokens
    print(f"{name}: tokens={n_tokens} loss/token={per_token:.4f} "
          f"perplexity={math.exp(per_token):.2f} bpb={total_nats / (n_bytes * LN2):.3f}")

# Worked example: multiple choice.
logprobs = np.array([-6.0, -9.0, -12.0])
lengths = np.array([3, 6, 12])
print("raw", logprobs, "argmax", int(np.argmax(logprobs)))
print("normalised", logprobs / lengths, "argmax", int(np.argmax(logprobs / lengths)))

# Worked example: pass@k.
n, c, k = 5, 2, 2
unbiased = 1 - math.comb(n - c, k) / math.comb(n, k)
with_replacement = 1 - (1 - c / n) ** k
print(f"pass@{k} n={n} c={c}: unbiased={unbiased:.3f} with-replacement={with_replacement:.3f} c/n={c / n:.3f}")
rng = np.random.default_rng(0)
hits = sum(rng.permutation(n)[:k].min() < c for _ in range(100000))  # first c indices are correct
print(f"Monte Carlo of drawing {k} of {n} without replacement: {hits / 100000:.3f}")

# Figure 2: two models on the same 200 items, B equals A except for 15 items.
a = np.zeros(200)
a[:120] = 1.0
b = a.copy()
b[:12] = 0.0          # A right, B wrong on 12 items
b[120:123] = 1.0      # A wrong, B right on 3 items
rng = np.random.default_rng(0)
B = 4000
idx = rng.integers(0, 200, size=(B, 200))
ma, mb, md = a[idx].mean(1), b[idx].mean(1), (b - a)[idx].mean(1)
for name, m, x in (("A", ma, a), ("B", mb, b), ("B minus A paired", md, b - a)):
    lo, hi = np.percentile(m, [2.5, 97.5])
    print(f"{name}: mean={x.mean():.4f} CI=({lo:.4f}, {hi:.4f})")
ia = rng.integers(0, 200, size=(B, 200))
ib = rng.integers(0, 200, size=(B, 200))
ud = b[ib].mean(1) - a[ia].mean(1)
lo, hi = np.percentile(ud, [2.5, 97.5])
print(f"B minus A independent resampling: CI=({lo:.4f}, {hi:.4f})")

# Contamination worked example.
test = "a b c d e".split()
train = "x b c d y".split()
tri = [tuple(test[i:i + 3]) for i in range(3)]
trs = {tuple(train[i:i + 3]) for i in range(3)}
print("trigram overlap", sum(t in trs for t in tri) / len(tri))
