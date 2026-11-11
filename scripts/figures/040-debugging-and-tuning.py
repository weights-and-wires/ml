#!/usr/bin/env python3
"""Numbers for the debugging-and-tuning figures.

Figure 1 is the class probabilities of an untrained four-way guess and of a
guess whose scores started too large. Figure 2 is the learning-rate bowl:
forty full-batch steps of one seeded network, restarted from the same weights
at each rate.

The lesson SVG uses the coordinates printed under "bar heights" and
"bowl polyline". Re-run this file before changing either figure.
"""

from __future__ import annotations

import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """Mean cross-entropy and the gradient of that mean with respect to logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1))
    n = logits.shape[0]
    loss = float(-(shifted[np.arange(n), y] - log_z).mean())
    probs = np.exp(shifted - log_z[:, None])
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, probs, grad


class MLP:
    """Two-layer ReLU classifier. The lesson's starter ships the same step."""

    def __init__(self, n_in, n_hidden, n_classes, rng, scale):
        self.n_classes = int(n_classes)
        self.W1 = rng.normal(scale=scale, size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(scale=scale, size=(n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)

    def loss_and_grads(self, X, y):
        pre = X @ self.W1 + self.b1
        hidden = np.maximum(pre, 0.0)
        logits = hidden @ self.W2 + self.b2
        loss, probs, dlogits = softmax_cross_entropy(logits, y)
        dhidden = dlogits @ self.W2.T * (pre > 0.0)
        grads = {
            "W1": X.T @ dhidden,
            "b1": dhidden.sum(axis=0),
            "W2": hidden.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        return loss, grads

    def sgd_step(self, grads, lr):
        for name in ("W1", "b1", "W2", "b2"):
            setattr(self, name, getattr(self, name) - lr * grads[name])


def bowl_data(seed=0, n=40):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def fresh_model():
    return MLP(2, 8, 2, np.random.default_rng(1), scale=0.1)


def range_losses(X, y, lrs, steps=40):
    losses = []
    for lr in lrs:
        model = fresh_model()
        final = np.inf
        for _ in range(steps):
            loss, grads = model.loss_and_grads(X, y)
            if not np.isfinite(loss):
                final = np.inf
                break
            model.sgd_step(grads, float(lr))
        else:
            final, _ = model.loss_and_grads(X, y)
            if not np.isfinite(final):
                final = np.inf
        losses.append(float(final))
    return losses


def bowl_point(lr, loss):
    """Map a rate and a loss onto the figure's 640 by 300 view."""
    x = 78 + (np.log10(lr) + 3) / 4 * 532
    y = 248 - loss * 200
    return round(float(x), 1), round(float(y), 1)


def main() -> None:
    print("uniform loss", float(np.log(4)))
    spiked = np.array([[4.0, 0.0, 0.0, 0.0]])
    loss, probs, _ = softmax_cross_entropy(spiked, np.array([2]))
    print("spiked loss", loss)
    print("spiked probs", probs[0])
    print("bar heights, probability times 150, baseline y=200")
    for name, row in ("uniform", np.full(4, 0.25)), ("spiked", probs[0]):
        for index, prob in enumerate(row):
            height = float(prob) * 150
            print(f"  {name} class {index}: h={height:.2f} y={200 - height:.2f}")

    X, y = bowl_data()
    lrs = np.logspace(-3, 1, 9)
    losses = range_losses(X, y, lrs, steps=40)
    print("bowl")
    points = []
    for lr, loss in zip(lrs, losses):
        point = bowl_point(lr, loss)
        points.append(point)
        print(f"  {lr:.8g} {loss:.8g} -> {point[0]},{point[1]}")
    print("polyline", " ".join(f"{x},{y}" for x, y in points))
    print("log 2 line y", bowl_point(1e-3, float(np.log(2)))[1])

    model = fresh_model()
    loss0, grads = model.loss_and_grads(X, y)
    print("init loss", loss0, "gap", abs(loss0 - np.log(2)))
    for name in ("W1", "W2"):
        base = np.linalg.norm(grads[name]) / np.linalg.norm(getattr(model, name))
        print(name, "base", base, "at 0.01", 0.01 * base, "at 3.16227766", 3.16227766 * base)

    wide = MLP(2, 8, 2, np.random.default_rng(1), scale=2.0)
    wide_loss, _ = wide.loss_and_grads(X, y)
    print("scale 2 loss", wide_loss)


if __name__ == "__main__":
    main()
