"""Softmax and cross-entropy with a readable gradient.

The Tensor engine from the previous chapters is provided complete. Fill in
softmax, log_softmax, and softmax_cross_entropy.
"""

import numpy as np


# Written in 'The backward rule for each array operation'.
def unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """The engine from the previous two chapters, provided complete."""

    def __init__(self, value, children=()):
        self.value = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.value)
        self._children = children
        self._backward_fn = lambda: None

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    # Written in 'Walking slopes backwards through a calculation'.
    def __add__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.value.shape)
            other.grad += unbroadcast(out.grad, other.value.shape)

        out._backward_fn = _backward
        return out

    # Written in 'Walking slopes backwards through a calculation'.
    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.value, self.value.shape)
            other.grad += unbroadcast(out.grad * self.value, other.value.shape)

        out._backward_fn = _backward
        return out

    # Written in 'The backward rule for each array operation'.
    def __matmul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value @ other.value, children=(self, other))

        def _backward():
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad

        out._backward_fn = _backward
        return out

    # Written in 'The backward rule for each array operation'.
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

    # Written in 'Walking slopes backwards through a calculation'.
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


if __name__ == "__main__":
    z = np.array([[2.0, 1.0, 0.0]])
    print("softmax:", np.round(softmax(z), 6).tolist())
    logits = Tensor(z)
    loss = softmax_cross_entropy(logits, np.array([0]))
    loss.backward()
    print(f"loss for class 0: {float(loss.value):.6f}")
    print("slope on the logits:", np.round(logits.grad, 6).tolist())
