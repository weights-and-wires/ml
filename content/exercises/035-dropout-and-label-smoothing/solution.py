"""Reference solution for the dropout and label-smoothing exercise."""

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def dropout_mask(shape, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(shape) < 1.0 - p).astype(float)


def dropout_forward(x: np.ndarray, p: float, rng: np.random.Generator, training: bool):
    if not training:
        return x, np.ones_like(x)
    mask = dropout_mask(x.shape, p, rng)
    out = x * mask / (1.0 - p)
    return out, mask


def dropout_backward(dout: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    return dout * mask / (1.0 - p)


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
    x = np.array([2.0, 4.0, 6.0, 8.0])
    rng = np.random.default_rng(0)

    out, mask = dropout_forward(x, 0.5, rng, training=True)
    print("mask:", mask, "out:", out)

    sums = np.array([dropout_forward(x, 0.5, rng, training=True)[0].sum() for _ in range(2000)])
    print(f"mean layer sum over 2000 masks: {sums.mean():.2f} (test time gives {x.sum():.0f})")

    fixed = np.array([1.0, 0.0, 1.0, 1.0])
    print("backward for dout = ones:", dropout_backward(np.ones(4), fixed, 0.5))

    logits = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    print("smoothed target:", np.round(smooth_targets(targets, 3, 0.1), 4))
    print(f"loss with smoothing {label_smoothing_loss(logits, targets, 0.1):.4f}, without {label_smoothing_loss(logits, targets, 0.0):.4f}")
    print("gradient:", np.round(label_smoothing_grad(logits, targets, 0.1), 4))

    gaps = np.linspace(0.0, 8.0, 801)
    losses = [label_smoothing_loss(np.array([[g, 0.0, 0.0]]), targets, 0.1) for g in gaps]
    print(f"loss is lowest at a gap of {gaps[int(np.argmin(losses))]:.2f} (ln 28 = {np.log(28):.2f})")
