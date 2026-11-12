"""Numbers behind the figures in 077-text-embeddings-and-retrieval.

Prints the values that the lesson's three figures and worked examples draw.
Pass --svg to also print the figure markup. Uses only the standard library
and numpy.
"""

import math
import sys

import numpy as np

# Worked example 1: pooling one 3-token text, padded to length 4.
hidden = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
pad = np.array([[9.0, 9.0]])
right = np.vstack([hidden, pad])
left = np.vstack([pad, hidden])
print("mean over the 3 real tokens:", hidden.mean(axis=0))
print("mean over all 4 rows, right-padded (the bug):", right.mean(axis=0))
print("last real token:", hidden[-1], "unit:", hidden[-1] / np.linalg.norm(hidden[-1]))
print("row -1 of the right-padded matrix (the bug):", right[-1])
print("row -1 of the left-padded matrix:", left[-1])

# Figure 1: how many tokens each position attends to under a causal mask.
T = 5
mask = np.tril(np.ones((T, T), dtype=int))
print("tokens seen per position:", mask.sum(axis=1).tolist())

# Worked example 2: InfoNCE with in-batch and hard negatives.
q = np.array([[1.0, 0.0], [0.0, 1.0]])
pos = np.array([[0.8, 0.6], [0.6, 0.8]])
hard = np.array([[0.6, -0.8], [0.8, -0.6]])
docs = np.vstack([pos, hard])


def info_nce(q, docs, tau):
    logits = q @ docs.T / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(q)), np.arange(len(q))]).mean(), p


for tau in (0.1, 0.02):
    full, p = info_nce(q, docs, tau)
    inb, _ = info_nce(q, pos, tau)
    print(f"tau={tau}: with hard negatives {full:.4f}, in-batch only {inb:.4f}")
    if tau == 0.1:
        print("softmax rows at tau=0.1:\n", np.round(p, 3))
print("log 4 =", round(math.log(4), 4), " log 2 =", round(math.log(2), 4))

# Worked example 3: ranking a toy corpus with graded relevance.
names = ["A", "B", "C", "D", "E", "F"]
rel = np.array([0, 3, 0, 2, 1, 0])
cos = np.array([0.95, 0.90, 0.80, 0.60, 0.30, -0.20])
order = np.argsort(-cos, kind="stable")
ranked = rel[order]
disc = 1.0 / np.log2(np.arange(2, len(rel) + 2))
for k in (3, 5):
    dcg = float((ranked[:k] * disc[:k]).sum())
    ideal = np.sort(rel)[::-1]
    idcg = float((ideal[:k] * disc[:k]).sum())
    found = int((ranked[:k] > 0).sum())
    print(f"k={k}: DCG={dcg:.4f} IDCG={idcg:.4f} nDCG={dcg / idcg:.4f} "
          f"recall={found}/{int((rel > 0).sum())}")
print("discounts:", np.round(disc, 4).tolist())
print("gain*discount at ranks 1-3:", np.round(ranked[:3] * disc[:3], 4).tolist())
print("ideal gain*discount at ranks 1-3:", np.round(np.sort(rel)[::-1][:3] * disc[:3], 4).tolist())
print("nDCG@3 =", round(float((ranked[:3] * disc[:3]).sum() / (np.sort(rel)[::-1][:3] * disc[:3]).sum()), 4))
