"""Gradients for a whole network, and the check that proves them.

The Tensor engine, central_difference, and sgd_step are provided. Fill in the
six functions below, in order: softmax, log_softmax, softmax_cross_entropy,
mlp_loss_and_grad, numeric_grad, and gradient_check.
"""

import numpy as np


# Provided.
def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


# Provided.
def unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """The engine from the previous chapter, provided complete."""

    def __init__(self, value, children=()):
        self.value = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.value)
        self._children = children
        self._backward_fn = lambda: None

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.value.shape)
            other.grad += unbroadcast(out.grad, other.value.shape)

        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.value, self.value.shape)
            other.grad += unbroadcast(out.grad * self.value, other.value.shape)

        out._backward_fn = _backward
        return out

    def __matmul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value @ other.value, children=(self, other))

        def _backward():
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad

        out._backward_fn = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.value, 0.0), children=(self,))

        def _backward():
            self.grad += out.grad * (self.value > 0.0)

        out._backward_fn = _backward
        return out

    def sum(self):
        out = Tensor(self.value.sum(), children=(self,))

        def _backward():
            self.grad += out.grad * np.ones_like(self.value)

        out._backward_fn = _backward
        return out

    def backward(self):
        order = []
        seen = set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node._children:
                visit(child)
            order.append(node)

        visit(self)
        self.grad = np.ones_like(self.value)
        for node in reversed(order):
            node._backward_fn()


def softmax(z):
    """Row-wise softmax that does not overflow.

    Args:
        z: An array of shape (n, c), one row of scores per example.

    Returns:
        An array of shape (n, c) whose rows are positive and sum to 1.
        Subtract each row's maximum before exponentiating. The shift cancels
        in the ratio, so no probability changes, but exp never sees a
        positive number.
    """
    raise NotImplementedError("softmax")


def log_softmax(z):
    """Row-wise log of softmax, without ever forming a tiny probability.

    Args:
        z: An array of shape (n, c).

    Returns:
        An array of shape (n, c) equal to log(softmax(z)), computed as
        (z - m) - log(sum(exp(z - m))) for the row maximum m. It stays
        finite at logits of plus or minus 1000.
    """
    raise NotImplementedError("log_softmax")


def softmax_cross_entropy(logits, labels):
    """Mean cross-entropy of softmax(logits) against integer labels.

    Args:
        logits: A Tensor of shape (n, c).
        labels: An integer array of shape (n,) with values in range(c). Plain
            data, not a Tensor.

    Returns:
        A scalar Tensor equal to -mean_i log_softmax(logits)[i, labels[i]],
        whose only child is `logits`. Its `_backward_fn` adds
        out.grad * (softmax(logits) - one_hot(labels)) / n into logits.grad.
    """
    raise NotImplementedError("softmax_cross_entropy")


def mlp_loss_and_grad(params, X, y):
    """Compute the loss and the gradient of every weight and bias in one backward pass.

    Args:
        params: A list of layers, each {"W": array (in, out), "b": array (out,)}.
            Every layer but the last is followed by relu. The last layer's
            output, the logits, goes straight to softmax_cross_entropy.
        X: Inputs, shape (n, d).
        y: Integer labels, shape (n,).

    Returns:
        (loss, grads): the loss as a float, and a list matching params whose
        entries {"W": array, "b": array} hold the gradients, each with the
        shape of its parameter. Build fresh Tensors on every call.
    """
    raise NotImplementedError("mlp_loss_and_grad")


def numeric_grad(loss_fn, array, h=1e-5):
    """Estimate the gradient of loss_fn at an array, one entry at a time.

    Args:
        loss_fn: A function that takes an array of the same shape as `array`
            and returns a float.
        array: The point to measure at. Do not change the caller's array.
        h: The nudge size.

    Returns:
        An array of the same shape holding, for each entry, the two-sided
        slope (loss_fn(x + h) - loss_fn(x - h)) / (2 h) with only that entry
        nudged.
    """
    raise NotImplementedError("numeric_grad")


def gradient_check(params, X, y):
    """Return the worst disagreement between backprop and nudging.

    For each parameter array, compare the analytic gradient from
    mlp_loss_and_grad with numeric_grad of the loss. Score each array as
    norm(analytic - numeric) / (norm(analytic) + norm(numeric)), taking 0
    when both are zero, and return the largest score as a float.
    """
    raise NotImplementedError("gradient_check")


# Provided: one step downhill. The next chapters build better steps.
def sgd_step(params, grads, lr):
    return [
        {"W": p["W"] - lr * g["W"], "b": p["b"] - lr * g["b"]}
        for p, g in zip(params, grads)
    ]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d, hidden, classes = 8, 4, 5, 3
    X = rng.normal(size=(n, d))
    y = rng.integers(classes, size=n)
    params = [
        {"W": rng.normal(size=(d, hidden)) * 0.5, "b": np.zeros(hidden)},
        {"W": rng.normal(size=(hidden, classes)) * 0.5, "b": np.zeros(classes)},
    ]
    loss, grads = mlp_loss_and_grad(params, X, y)
    print(f"loss: {loss:.4f}")
    print(f"gradient check, worst relative error: {gradient_check(params, X, y):.2e}")
    for step in range(5):
        params = sgd_step(params, grads, lr=0.5)
        loss, grads = mlp_loss_and_grad(params, X, y)
        print(f"after step {step + 1}: loss {loss:.4f}")
