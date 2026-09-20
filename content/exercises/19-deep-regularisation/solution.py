"""Reference solution for the deep-regularisation exercise."""

import numpy as np


def dropout_forward(x: np.ndarray, p: float, rng: np.random.Generator, training: bool):
    if not training:
        return x, np.ones_like(x)
    keep = 1.0 - p
    mask = (rng.random(x.shape) < keep).astype(x.dtype)
    out = x * mask / keep
    return out, mask


def dropout_backward(dout: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    keep = 1.0 - p
    return dout * mask / keep


def weight_decay_step(w: np.ndarray, grad: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    return w - lr * grad - lr * weight_decay * w


def _smoothed_targets(logits: np.ndarray, targets: np.ndarray, eps: float) -> np.ndarray:
    n, k = logits.shape
    one_hot = np.eye(k)[targets]
    return (1.0 - eps) * one_hot + eps / k


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_z


def label_smoothing_loss(logits: np.ndarray, targets: np.ndarray, eps: float) -> float:
    y_smooth = _smoothed_targets(logits, targets, eps)
    log_p = _log_softmax(logits)
    per_row = -(y_smooth * log_p).sum(axis=1)
    return float(per_row.mean())


def label_smoothing_grad(logits: np.ndarray, targets: np.ndarray, eps: float) -> np.ndarray:
    n = logits.shape[0]
    y_smooth = _smoothed_targets(logits, targets, eps)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    p = exp / exp.sum(axis=1, keepdims=True)
    return (p - y_smooth) / n


def early_stopping_equivalent_ridge(eigenvalue: float, step: float, iterations: int) -> float:
    if iterations == 0:
        return float("inf")
    base = (1.0 - step * eigenvalue) ** iterations
    denom = 1.0 - base
    if denom <= 0.0:
        return 0.0
    return eigenvalue * base / denom


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.ones(200_000)
    out, mask = dropout_forward(x, p=0.4, rng=rng, training=True)
    print(f"dropout: kept fraction {mask.mean():.3f}, mean output {out.mean():.3f} (should track 1.0)")

    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    targets = np.array([0, 1])
    loss_smoothed = label_smoothing_loss(logits, targets, eps=0.1)
    loss_plain = label_smoothing_loss(logits, targets, eps=0.0)
    print(f"loss with smoothing: {loss_smoothed:.4f}, without: {loss_plain:.4f}")

    for k in (0, 1, 10, 1000):
        gamma = early_stopping_equivalent_ridge(eigenvalue=2.0, step=0.1, iterations=k)
        print(f"iterations={k:>5}: equivalent ridge penalty = {gamma:.4g}")
