#!/usr/bin/env python3
"""Numbers for the scoring-classifiers figures.

Figure 2 is the ROC of the eight loans in the lesson. Figure 3 is a
reliability diagram for 500 seeded rows whose log-odds were multiplied by 3,
before and after the temperature that minimises the log loss.

The lesson SVG uses the coordinates printed under "ROC polyline" and
"reliability points". Re-run this file before changing either figure.
"""

from __future__ import annotations

import numpy as np


LOAN_SCORES = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30, 0.25, 0.10])
LOAN_Y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def roc_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    scores = np.asarray(scores, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    fpr = [0.0]
    tpr = [0.0]
    tp = 0.0
    fp = 0.0
    index = 0
    while index < len(y):
        end = index
        while end < len(y) and scores[end] == scores[index]:
            if y[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
        index = end
    return np.array(fpr), np.array(tpr)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    diff = pos[:, None] - neg[None, :]
    wins = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    return float(wins.mean())


def trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def ece(probs: np.ndarray, y: np.ndarray, bins: int) -> float:
    index = np.minimum(np.floor(probs * bins).astype(int), bins - 1)
    total = 0.0
    n = len(y)
    for bin_id in range(bins):
        mask = index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(y[mask].mean()) - float(probs[mask].mean()))
        total += (count / n) * gap
    return total


def temperature_grad(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    u = logits / temperature
    p = sigmoid(u)
    return float(np.sum((p - y) * (-logits / temperature**2)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    temperature = 1.0
    eps = 1e-4
    for _ in range(40):
        slope = temperature_grad(logits, y, temperature)
        if abs(slope) < 1e-8:
            break
        high = temperature_grad(logits, y, temperature + eps)
        low = temperature_grad(logits, y, temperature - eps)
        bend = (high - low) / (2 * eps)
        step = slope / bend
        while temperature - step <= 1e-8:
            step *= 0.5
        temperature = temperature - step
    return float(temperature)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    u = logits / temperature
    return float(np.sum(softplus(u) - y * u))


def reliability(probs: np.ndarray, y: np.ndarray, bins: int) -> list[tuple[int, int, float, float]]:
    index = np.minimum(np.floor(np.clip(probs, 0.0, 1.0) * bins).astype(int), bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append((bin_id, count, float(probs[mask].mean()), float(y[mask].mean())))
    return rows


def main() -> None:
    fpr, tpr = roc_curve(LOAN_SCORES, LOAN_Y)
    print("loan auc", auc(LOAN_SCORES, LOAN_Y), "trapezoid", trapezoid(fpr, tpr))
    print("loan ece 4 bins", ece(LOAN_SCORES, LOAN_Y, 4))
    left, bottom, size = 64.0, 288.0, 240.0
    pts = []
    for x_rate, y_rate in zip(fpr, tpr):
        pts.append(f"{left + x_rate * size:.1f},{bottom - y_rate * size:.1f}")
    print("ROC polyline", " ".join(pts))

    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    y = (rng.uniform(size=500) < sigmoid(z)).astype(float)
    logits = 3.0 * z
    temperature = fit_temperature(logits, y)
    raw = sigmoid(logits)
    scaled = sigmoid(logits / temperature)
    print("temperature", temperature)
    print("nll T=1", nll(logits, y, 1.0), "nll fitted", nll(logits, y, temperature))
    print("ece10 raw", ece(raw, y, 10), "ece10 scaled", ece(scaled, y, 10))
    print("one-row loss", float(softplus(np.array(1.0)) - 1.0))

    plot_left, plot_bottom, plot_size = 72.0, 284.0, 260.0
    print("reliability before")
    for bin_id, count, conf, acc in reliability(raw, y, 8):
        x = plot_left + conf * plot_size
        y_px = plot_bottom - acc * plot_size
        print(f"  bin {bin_id} n={count} conf={conf:.4f} acc={acc:.4f} xy={x:.1f},{y_px:.1f}")
    print("reliability after")
    for bin_id, count, conf, acc in reliability(scaled, y, 8):
        x = plot_left + conf * plot_size
        y_px = plot_bottom - acc * plot_size
        print(f"  bin {bin_id} n={count} conf={conf:.4f} acc={acc:.4f} xy={x:.1f},{y_px:.1f}")


if __name__ == "__main__":
    main()
