#!/usr/bin/env python3
"""Numbers for the counts-to-a-language-model figures.

Figure 1 is the five-word tally in the lesson, before and after add-1
smoothing. Figure 2 is the cosine of each center vector with dog, after
the seeded skip-gram in the lesson (20 passes, learning rate 0.4, 4
negative samples, dimension 8, seed 0).

Re-run this file before changing either figure. The bars use the
coordinates printed under "figure 2 rects".
"""

from __future__ import annotations

import math

import numpy as np


HAND_PREV = np.array([0, 1, 2, 0, 1, 3])
HAND_NXT = np.array([1, 2, 4, 1, 3, 4])
HAND_V = 5
HAND_K = 1.0

# Training notebook. Indices: start big small dog hound cat kitten runs sits.
BASE = np.array(
    [
        [0, 1, 3, 7],
        [0, 1, 4, 7],
        [0, 2, 5, 8],
        [0, 2, 6, 8],
    ],
    dtype=int,
)
TRAIN = np.repeat(BASE, 4, axis=0)
TRAIN_V = 9


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def bigram_counts(prev: np.ndarray, nxt: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size), dtype=float)
    np.add.at(counts, (prev, nxt), 1.0)
    return counts


def add_k_probs(counts: np.ndarray, k: float) -> np.ndarray:
    smoothed = np.asarray(counts, dtype=float) + k
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def pairs(sentences: np.ndarray, window: int) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for sentence in sentences:
        length = len(sentence)
        for center_at in range(length):
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                context_at = center_at + offset
                if 0 <= context_at < length:
                    found.append((int(sentence[center_at]), int(sentence[context_at])))
    return found


def unigram_noise(sentences: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.bincount(sentences.ravel(), minlength=vocab_size).astype(float)
    weights = np.power(counts, 0.75)
    return weights / weights.sum()


def train_skipgram(
    sentences: np.ndarray,
    vocab_size: int,
    window: int,
    dim: int,
    epochs: int,
    lr: float,
    n_negative: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    center = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    context = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    pair_list = pairs(sentences, window)
    noise = unigram_noise(sentences, vocab_size)
    for _ in range(epochs):
        order = rng.permutation(len(pair_list))
        for index in order:
            center_id, positive_id = pair_list[int(index)]
            negatives: list[int] = []
            while len(negatives) < n_negative:
                draw = int(rng.choice(vocab_size, p=noise))
                if draw != positive_id:
                    negatives.append(draw)
            negative_ids = np.asarray(negatives, dtype=int)
            center_vec = center[center_id].copy()
            positive_vec = context[positive_id].copy()
            negative_vecs = context[negative_ids].copy()
            positive_score = float(center_vec @ positive_vec)
            negative_scores = negative_vecs @ center_vec
            positive_gate = float(sigmoid(np.asarray(positive_score)) - 1.0)
            negative_gates = sigmoid(negative_scores)
            grad_center = positive_gate * positive_vec + (negative_gates[:, None] * negative_vecs).sum(axis=0)
            grad_positive = positive_gate * center_vec
            grad_negatives = negative_gates[:, None] * center_vec
            center[center_id] = center_vec - lr * grad_center
            context[positive_id] = positive_vec - lr * grad_positive
            for slot, negative_id in enumerate(negative_ids):
                context[negative_id] = context[negative_id] - lr * grad_negatives[slot]
    return center


def cosine(table: np.ndarray, left: int, right: int) -> float:
    a = table[left]
    b = table[right]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def train_window_lm(
    sentences: np.ndarray,
    vocab_size: int,
    window: int,
    dim: int,
    hidden: int,
    steps: int,
    lr: float,
    seed: int,
    query_context: np.ndarray,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    table = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    weight_1 = rng.normal(0.0, 0.1, size=(window * dim, hidden))
    bias_1 = np.zeros(hidden)
    weight_2 = rng.normal(0.0, 0.1, size=(hidden, vocab_size))
    bias_2 = np.zeros(vocab_size)
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    for sentence in sentences:
        for index in range(window, len(sentence)):
            contexts.append(sentence[index - window : index])
            targets.append(int(sentence[index]))
    context_ids = np.asarray(contexts, dtype=int)
    target_ids = np.asarray(targets, dtype=int)
    for _ in range(steps):
        hidden_in = table[context_ids].reshape(len(context_ids), -1)
        hidden_pre = hidden_in @ weight_1 + bias_1
        hidden_act = np.tanh(hidden_pre)
        logits = hidden_act @ weight_2 + bias_2
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        grad_logits = probs.copy()
        grad_logits[np.arange(len(target_ids)), target_ids] -= 1.0
        grad_logits /= len(target_ids)
        grad_weight_2 = hidden_act.T @ grad_logits
        grad_bias_2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ weight_2.T * (1.0 - hidden_act**2)
        grad_weight_1 = hidden_in.T @ grad_hidden
        grad_bias_1 = grad_hidden.sum(axis=0)
        grad_embed_in = (grad_hidden @ weight_1.T).reshape(len(context_ids), window, dim)
        grad_table = np.zeros_like(table)
        np.add.at(grad_table, context_ids, grad_embed_in)
        table -= lr * grad_table
        weight_1 -= lr * grad_weight_1
        bias_1 -= lr * grad_bias_1
        weight_2 -= lr * grad_weight_2
        bias_2 -= lr * grad_bias_2
    query = np.asarray(query_context, dtype=int)
    hidden_in = table[query].reshape(1, -1)
    hidden_act = np.tanh(hidden_in @ weight_1 + bias_1)
    logits = (hidden_act @ weight_2 + bias_2).ravel()
    shifted = logits - logits.max()
    dist = np.exp(shifted)
    dist /= dist.sum()
    hidden_in = table[context_ids].reshape(len(context_ids), -1)
    hidden_act = np.tanh(hidden_in @ weight_1 + bias_1)
    logits = hidden_act @ weight_2 + bias_2
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    nll = float(-np.mean(np.log(probs[np.arange(len(target_ids)), target_ids])))
    return nll, dist


def main() -> None:
    counts = bigram_counts(HAND_PREV, HAND_NXT, HAND_V)
    probs = add_k_probs(counts, HAND_K)
    pair_p = probs[HAND_PREV, HAND_NXT]
    nll = float(-np.mean(np.log(pair_p)))
    print("counts")
    print(counts.astype(int))
    print("probs")
    print(np.array2string(probs, precision=6))
    print(f"nll {nll:.10f} perplexity {math.exp(nll):.10f}")

    center = np.array([1.0, 0.0])
    positive = np.array([1.0, 0.0])
    negative = np.array([[0.0, 1.0]])
    positive_score = float(center @ positive)
    negative_score = float((negative @ center).reshape(-1)[0])
    loss = float(softplus(np.asarray(-positive_score)) + softplus(np.asarray(negative_score)))
    print(f"sigmoid(1) {float(sigmoid(np.asarray(positive_score))):.10f}")
    print(f"tiny loss {loss:.10f}")

    table = train_skipgram(TRAIN, TRAIN_V, 1, 8, 20, 0.4, 4, 0)
    names = {
        "hound": 4,
        "cat": 5,
        "runs": 7,
        "sits": 8,
    }
    print("cosines with dog")
    baseline = 248.0
    top = 40.0
    span = baseline - top
    # Four bars, left edges. Plot origin x = 110, bar width 80, gap 36.
    lefts = [110, 226, 342, 458]
    for left, (name, index) in zip(lefts, names.items()):
        value = cosine(table, 3, index)
        y = baseline - value * span
        height = value * span
        print(
            f"  {name} cosine {value:.6f} rect x={left} y={y:.1f} "
            f"width=80 height={height:.1f}"
        )

    nll_lm, dist = train_window_lm(TRAIN, TRAIN_V, 2, 8, 16, 80, 0.5, 0, np.array([1, 3]))
    print(
        f"window nll {nll_lm:.6f} P(runs) {dist[7]:.6f} P(sits) {dist[8]:.6f} "
        f"optimal {(math.log(2) / 2):.6f} lnV {math.log(TRAIN_V):.6f}"
    )


if __name__ == "__main__":
    main()
