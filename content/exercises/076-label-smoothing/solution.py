"""Reference solution for the label-smoothing exercise."""

import numpy as np


# Provided: the max-shifted log-softmax, row by row.
def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def smooth_targets(targets: np.ndarray, num_classes: int, eps: float) -> np.ndarray:
    return (1.0 - eps) * np.eye(num_classes)[targets] + eps / num_classes


def label_smoothing_loss(logits: np.ndarray, targets: np.ndarray, eps: float) -> float:
    y_smooth = smooth_targets(targets, logits.shape[1], eps)
    per_row = -(y_smooth * log_softmax(logits)).sum(axis=1)
    return float(per_row.mean())


def label_smoothing_grad(logits: np.ndarray, targets: np.ndarray, eps: float) -> np.ndarray:
    n, k = logits.shape
    y_smooth = smooth_targets(targets, k, eps)
    p = np.exp(log_softmax(logits))
    return (p - y_smooth) / n


if __name__ == "__main__":
    logits = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    print("smoothed target:", np.round(smooth_targets(targets, 3, 0.1), 4))
    print(f"loss with smoothing {label_smoothing_loss(logits, targets, 0.1):.4f}, without {label_smoothing_loss(logits, targets, 0.0):.4f}")
    print("gradient:", np.round(label_smoothing_grad(logits, targets, 0.1), 4))

    gaps = np.linspace(0.0, 8.0, 801)
    losses = [label_smoothing_loss(np.array([[g, 0.0, 0.0]]), targets, 0.1) for g in gaps]
    print(f"loss is lowest at a gap of {gaps[int(np.argmin(losses))]:.2f} (ln 28 = {np.log(28):.2f})")
