"""Diagnostics for a tiny two-layer classifier.

The network is given. You write the checks that say whether a run is worth
trusting: the loss before any training, a learning-rate sweep, a log-uniform
search, and the size of one step relative to the weights.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """A two-layer classifier: linear, ReLU, linear, mean cross-entropy.

    ``W1`` has shape ``(n_in, n_hidden)`` and ``W2`` has shape
    ``(n_hidden, n_classes)``. Biases start at zero. ``scale`` is the standard
    deviation of the initial weights.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_classes: int,
        rng: np.random.Generator,
        scale: float = 0.1,
    ) -> None:
        self.n_classes = int(n_classes)
        self.W1 = rng.normal(scale=scale, size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(scale=scale, size=(n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)

    def loss_and_grads(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Return the mean cross-entropy and the gradients of that mean.

        ``X`` has shape ``(n, n_in)`` and ``y`` has shape ``(n,)`` with integer
        labels in ``0 .. n_classes - 1``. The gradient dict has ``W1``, ``b1``,
        ``W2``, and ``b2``, each the same shape as the matching parameter.
        """
        pre = X @ self.W1 + self.b1
        hidden = np.maximum(pre, 0.0)
        logits = hidden @ self.W2 + self.b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_z = np.log(np.exp(shifted).sum(axis=1))
        n = logits.shape[0]
        loss = float(-(shifted[np.arange(n), y] - log_z).mean())
        probs = np.exp(shifted - log_z[:, None])
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        dhidden = dlogits @ self.W2.T * (pre > 0.0)
        grads = {
            "W1": X.T @ dhidden,
            "b1": dhidden.sum(axis=0),
            "W2": hidden.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        return loss, grads

    def sgd_step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """Subtract ``lr`` times each gradient from the matching parameter."""
        self.W1 = self.W1 - lr * grads["W1"]
        self.b1 = self.b1 - lr * grads["b1"]
        self.W2 = self.W2 - lr * grads["W2"]
        self.b2 = self.b2 - lr * grads["b2"]


def expected_initial_loss(C: int) -> float:
    """Return the cross-entropy of a uniform guess over ``C`` classes.

    That number is the natural log of ``C``.
    """
    raise NotImplementedError("write expected_initial_loss")


def check_initial_loss(model: MLP, X: np.ndarray, y: np.ndarray) -> bool:
    """Return whether the model's mean loss is within 0.5 of ``log(C)``.

    Read the loss from ``model.loss_and_grads`` and the class count from
    ``model.n_classes``. A loss of about ``10 * log(C)`` is a bug and returns
    false.
    """
    raise NotImplementedError("write check_initial_loss")


def lr_range_test(model_fn, X: np.ndarray, y: np.ndarray, lrs, steps: int = 40) -> np.ndarray:
    """Return the loss after ``steps`` full-batch updates at each learning rate.

    ``model_fn()`` builds a fresh network. Call it once per rate so every rate
    starts from that function's weights. After a non-finite loss, record
    ``inf`` for that rate and move on. The result has one float per rate, in
    the same order as ``lrs``.
    """
    raise NotImplementedError("write lr_range_test")


def pick_lr(lrs, losses) -> float:
    """Return the learning rate whose loss is the smallest finite one.

    A non-finite loss is not the pick. When several rates share the smallest
    loss, return the earliest one.
    """
    raise NotImplementedError("write pick_lr")


def random_search(objective, space: dict, n: int, seed: int) -> dict:
    """Draw ``n`` log-uniform configs and return the one with the lowest score.

    ``space`` maps a name to ``(low, high)``, both positive. Visit the names in
    sorted order. Each value is ``10 ** uniform(log10(low), log10(high))`` from
    ``np.random.default_rng(seed)``, one full config per trial.

    Returns a dict with ``samples`` (the configs in draw order), ``scores``
    (``objective`` on each, in that order), and ``best`` (the earliest config
    whose score is the smallest).
    """
    raise NotImplementedError("write random_search")


def update_ratio_stats(model: MLP, X: np.ndarray, y: np.ndarray, lr: float) -> dict:
    """Return ``||lr * gradient|| / ||weight||`` for ``W1`` and ``W2``.

    Use the gradients from ``loss_and_grads``. Do not change the weights. The
    keys are ``"W1"`` and ``"W2"``.
    """
    raise NotImplementedError("write update_ratio_stats")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    model = MLP(2, 4, 2, np.random.default_rng(1), scale=0.1)
    print("untrained loss for 2 classes:", expected_initial_loss(2))
    print("initial loss looks right:", check_initial_loss(model, X, y))
    rates = np.logspace(-2, 0, 5)

    def model_fn():
        return MLP(2, 4, 2, np.random.default_rng(1), scale=0.1)

    losses = lr_range_test(model_fn, X, y, rates, steps=20)
    print("losses:", losses)
    print("picked rate:", pick_lr(rates, losses))
    found = random_search(
        lambda config: config["lr"],
        {"lr": (1e-3, 1.0)},
        n=4,
        seed=0,
    )
    print("best sampled lr:", found["best"]["lr"])
    print("update ratios:", update_ratio_stats(model, X, y, 0.01))
