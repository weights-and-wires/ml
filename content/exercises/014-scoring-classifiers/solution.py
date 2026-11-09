"""Score a classifier: counts, an ROC curve, and a temperature."""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Count true positives, false positives, false negatives, and true negatives."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([tp, fp, fn, tn], dtype=float)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return precision, recall, and F1, with 0 when a denominator is 0."""
    tp, fp, fn, _tn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return np.array([precision, recall, f1], dtype=float)


def threshold_from_cost(cost_fp: float, cost_fn: float) -> float:
    """Return cost_fp / (cost_fp + cost_fn)."""
    return float(cost_fp) / (float(cost_fp) + float(cost_fn))


def roc_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trace the ROC, one point per distinct score, ties counted before the point."""
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    y = y[order]
    fpr = [0.0]
    tpr = [0.0]
    tp = 0.0
    fp = 0.0
    i = 0
    n = scores.size
    while i < n:
        score = scores[i]
        while i < n and scores[i] == score:
            if y[i] == 1:
                tp += 1.0
            else:
                fp += 1.0
            i += 1
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Return the pair average, with a tie worth 1/2."""
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    pos = scores[y == 1]
    neg = scores[y == 0]
    diff = pos[:, None] - neg[None, :]
    pair = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    return float(pair.mean())


def ece(probs: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Return the weighted mean absolute gap between rate and probability."""
    probs = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    index = np.minimum(np.floor(probs * bins).astype(int), bins - 1)
    total = 0.0
    n = probs.size
    for b in range(bins):
        mask = index == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        gap = abs(float(np.mean(y[mask])) - float(np.mean(probs[mask])))
        total += (count / n) * gap
    return float(total)


def temperature_nll_grad(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Return the slope of the summed log loss in the temperature."""
    logits = np.asarray(logits, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    temperature = float(temperature)
    p = _sigmoid(logits / temperature)
    return float(np.sum((p - y) * (-logits / temperature**2)))


def temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit a positive temperature by Newton, starting at 1."""
    temperature = 1.0
    eps = 1e-4
    for _ in range(40):
        slope = temperature_nll_grad(logits, y, temperature)
        if abs(slope) < 1e-8:
            break
        bend = (
            temperature_nll_grad(logits, y, temperature + eps)
            - temperature_nll_grad(logits, y, temperature - eps)
        ) / (2.0 * eps)
        step = slope / bend
        while temperature - step <= 1e-8:
            step *= 0.5
        temperature = temperature - step
    return float(temperature)


if __name__ == "__main__":
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30, 0.25, 0.10])
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    pred = (scores >= 0.5).astype(float)
    print("counts", confusion_counts(y, pred))
    print("precision, recall, f1", precision_recall_f1(y, pred))
    print("cost cut", threshold_from_cost(1.0, 9.0))
    fpr, tpr = roc_curve(scores, y)
    print("roc fpr", fpr)
    print("roc tpr", tpr)
    print("auc", auc(scores, y))
    print("ece", ece(scores, y, 4))
