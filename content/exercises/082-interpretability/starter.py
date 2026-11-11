"""Interpretability: probes, attribution, and sparse autoencoders.

Three tools for asking what a model computes. A linear probe reads a feature
out of frozen hidden states. Attribution (gradient times input, integrated
gradients, exact Shapley values) shares one output among the input features.
A sparse autoencoder pulls planted features back out of activations that
store more features than they have dimensions.

A model here is a Python function f that maps a vector of shape (d,) to one
number, and grad_f maps the same vector to the gradient, shape (d,).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- probes


def probe_loss_grad(
    w: np.ndarray, b: float, H: np.ndarray, y: np.ndarray, l2: float
) -> tuple[float, np.ndarray, float]:
    """Loss and gradient of a logistic probe.

    Args:
        w: probe weights, shape (d,).
        b: probe bias, a float.
        H: frozen hidden states, shape (n, d).
        y: labels in {0, 1}, shape (n,).
        l2: weight of the penalty 0.5 * l2 * ||w||^2 (the bias is not penalised).

    Returns:
        (loss, grad_w, grad_b): the mean logistic loss plus the penalty, the
        gradient on w with shape (d,), and the gradient on b as a float.
        The loss must stay finite when the scores are huge.
    """
    raise NotImplementedError("probe_loss_grad")


def train_probe(
    H: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.5, l2: float = 1e-3
) -> tuple[np.ndarray, float]:
    """Fit a logistic probe by full-batch gradient descent, from w = 0 and b = 0.

    Args:
        H: frozen hidden states, shape (n, d).
        y: labels in {0, 1}, shape (n,).
        steps: number of gradient steps.
        lr: step size.
        l2: weight of the penalty in probe_loss_grad.

    Returns:
        (w, b): the weights, shape (d,), and the bias, a float.
    """
    raise NotImplementedError("train_probe")


def probe_accuracy(w: np.ndarray, b: float, H: np.ndarray, y: np.ndarray) -> float:
    """Fraction of rows where the sign of H @ w + b matches the label.

    Args:
        w: probe weights, shape (d,).
        b: probe bias.
        H: hidden states, shape (n, d).
        y: labels in {0, 1}, shape (n,).

    Returns:
        A float in [0, 1]. A score above zero predicts label 1.
    """
    raise NotImplementedError("probe_accuracy")


# ----------------------------------------------------------- attribution


def grad_times_input(grad_f, x: np.ndarray) -> np.ndarray:
    """Gradient times input: grad_f(x) * x, entry by entry.

    Args:
        grad_f: function from a vector of shape (d,) to its gradient, shape (d,).
        x: the input, shape (d,).

    Returns:
        Shape (d,), one attribution per feature.
    """
    raise NotImplementedError("grad_times_input")


def integrated_gradients(f, grad_f, x: np.ndarray, baseline: np.ndarray, steps: int) -> np.ndarray:
    """Integrated gradients with the midpoint rule on the straight path.

    The path is baseline + a * (x - baseline) for a from 0 to 1. Split it into
    `steps` equal pieces and take the gradient at the middle of each piece,
    that is at a = (k + 0.5) / steps for k = 0 .. steps - 1.

    Args:
        f: the model. This function does not call it; it is part of the
            signature so a caller can pass the same pair to a completeness check.
        grad_f: gradient function, shape (d,) in and out.
        x: the input, shape (d,).
        baseline: the reference input, shape (d,).
        steps: number of pieces.

    Returns:
        Shape (d,): (x - baseline) times the average gradient along the path.
    """
    raise NotImplementedError("integrated_gradients")


def exact_shapley(f, x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Exact Shapley values by trying every subset of features.

    A subset S of features "present" means: take x on S and the baseline
    everywhere else. The value of S is f of that mixed input. Feature i gets
    the weighted average of its marginal contribution v(S + i) - v(S) over
    every subset S that excludes i, with weight |S|! (n - |S| - 1)! / n!.

    Args:
        f: the model, a vector of shape (n,) in and a float out. Use it for
            at most about 6 features, since it is called 2^n times.
        x: the input, shape (n,).
        baseline: the reference input, shape (n,).

    Returns:
        Shape (n,). The entries sum to f(x) - f(baseline).
    """
    raise NotImplementedError("exact_shapley")


# ------------------------------------------------------ sparse autoencoder


def sae_loss_grads(params: dict, X: np.ndarray, l1: float) -> tuple[float, dict]:
    """Loss and gradients of a sparse autoencoder.

    The model is f = relu(X @ We + be) and X_hat = f @ Wd. The loss is the
    mean over rows of ||X - X_hat||^2 plus l1 times the mean over rows of
    the sum of the entries of f.

    Args:
        params: dict with "We" of shape (d, k), "be" of shape (k,), and
            "Wd" of shape (k, d).
        X: activations, shape (n, d).
        l1: weight of the sparsity penalty.

    Returns:
        (loss, grads): a float and a dict with the same keys and shapes as params.
    """
    raise NotImplementedError("sae_loss_grads")


def sae_step(params: dict, X: np.ndarray, l1: float, lr: float) -> tuple[dict, float]:
    """One gradient step, then rescale every row of Wd to unit length.

    Args:
        params: the dict described in sae_loss_grads. Do not change it in place.
        X: activations, shape (n, d).
        l1: weight of the sparsity penalty.
        lr: step size.

    Returns:
        (new_params, loss): the updated dict and the loss before the step.
    """
    raise NotImplementedError("sae_step")


def train_sae(
    X: np.ndarray, k: int, l1: float = 0.3, lr: float = 0.2, steps: int = 4000, seed: int = 0
) -> dict:
    """Train a sparse autoencoder with k learned features.

    Start with Wd made of k random unit-length rows drawn with
    np.random.default_rng(seed), We = Wd.T, and be = 0, then take `steps`
    calls of sae_step.

    Args:
        X: activations, shape (n, d).
        k: number of features to learn.
        l1: weight of the sparsity penalty.
        lr: step size.
        steps: number of steps.
        seed: seed for the starting Wd.

    Returns:
        The trained params dict.
    """
    raise NotImplementedError("train_sae")


def match_score(Wd: np.ndarray, D: np.ndarray) -> float:
    """Mean over true directions of the best absolute cosine with a learned row.

    Args:
        Wd: learned feature directions, shape (k, d), any row lengths.
        D: true feature directions, shape (m, d), any row lengths.

    Returns:
        A float in [0, 1]. It is 1 when every true direction appears in Wd,
        up to sign and scale.
    """
    raise NotImplementedError("match_score")


if __name__ == "__main__":
    f = lambda v: v[0] * v[1] + 2 * v[2]
    x, base = np.array([2.0, 3.0, 1.0]), np.zeros(3)
    phi = exact_shapley(f, x, base)
    print("shapley:", phi, "sum:", phi.sum(), "f(x) - f(base):", f(x) - f(base))
